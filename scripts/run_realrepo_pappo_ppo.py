"""Run a staged PAPPO-PPO experiment on RealRepoFix."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.ppo_rollout import build_ppo_samples_from_trajectory  # noqa: E402
from pappo.ppo_training import (  # noqa: E402
    apply_advantage_prior,
    fill_samples_with_model_logprobs,
    train_pappo_ppo_update,
)
from pappo.turn_critic import GroupMeanTurnCritic, MeanTurnCritic  # noqa: E402
from scripts.run_realrepo_lora_comparison import (  # noqa: E402
    _make_backend,
    _rollout,
    _set_rollout_seed,
    _task_slice,
    _write_trajectories,
    evaluate_trajectories,
)


def _write_samples(path: Path, samples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), sort_keys=True) + "\n")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _grouped_train_rollout(args, adapter_path: Path | None):
    tasks = _task_slice(args.manifest, 0, args.train_limit, args.seed)
    backend = _make_backend(
        args.backend,
        args.model,
        adapter_path,
        args.max_new_tokens,
        args.temperature,
        args.local_files_only,
    )
    trajectories = []
    try:
        for task_index, task in enumerate(tasks):
            for rollout_index in range(args.num_rollouts_per_task):
                _set_rollout_seed(
                    args.seed,
                    task_index * args.num_rollouts_per_task + rollout_index,
                )
                trajectories.append(backend.repair(task))
        return trajectories
    finally:
        del backend
        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="tiny-local")
    parser.add_argument("--backend", choices=["scripted", "hf"], default="scripted")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=2)
    parser.add_argument("--eval-limit", type=int, default=1)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-rollouts-per-task", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--local-prior", type=float, default=0.25)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_eval = _rollout(
        backend_name=args.backend,
        model=args.model,
        adapter_path=None,
        manifest=args.manifest,
        start=args.train_limit,
        limit=args.eval_limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )
    base_metrics = evaluate_trajectories(base_eval)
    _write_trajectories(args.output_dir / "eval" / "base_rollouts.jsonl", base_eval)
    _write_json(args.output_dir / "eval" / "base_metrics.json", base_metrics)

    update_reports = []
    adapter_path = None
    for update_index in range(args.updates):
        update_dir = args.output_dir / f"update_{update_index:03d}"
        if args.num_rollouts_per_task > 1:
            train_rollouts = _grouped_train_rollout(args, adapter_path)
        else:
            train_rollouts = _rollout(
                backend_name=args.backend,
                model=args.model,
                adapter_path=adapter_path,
                manifest=args.manifest,
                start=0,
                limit=args.train_limit,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                local_files_only=args.local_files_only,
                seed=args.seed,
            )
        _write_trajectories(update_dir / "rollouts.jsonl", train_rollouts)
        samples = [
            sample
            for trajectory in train_rollouts
            for sample in build_ppo_samples_from_trajectory(trajectory)
        ]
        if args.num_rollouts_per_task > 1:
            group_critic = GroupMeanTurnCritic()
            group_critic.fit(
                group_keys=[sample.trajectory_id for sample in samples],
                tool_names=[sample.tool_name for sample in samples],
                targets=[sample.target for sample in samples],
            )
            samples = [
                replace(
                    sample,
                    value=group_critic.predict(sample.trajectory_id, sample.tool_name),
                )
                for sample in samples
            ]
            critic = group_critic
            critic_type = "group_mean"
        else:
            mean_critic = MeanTurnCritic()
            mean_critic.fit(
                tool_names=[sample.tool_name for sample in samples],
                targets=[sample.target for sample in samples],
            )
            samples = [
                replace(sample, value=mean_critic.predict(sample.tool_name))
                for sample in samples
            ]
            critic = mean_critic
            critic_type = "tool_mean"
        samples = apply_advantage_prior(samples, prior=args.local_prior)
        samples = fill_samples_with_model_logprobs(
            model_name=args.model,
            samples=samples,
            old_adapter_path=adapter_path,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
        )
        _write_samples(update_dir / "ppo_samples.jsonl", samples)
        critic_metrics = {
            "critic_type": critic_type,
            "advantage_prior": args.local_prior,
            "critic_samples": len(samples),
            "critic_value_mean": (
                sum(sample.value for sample in samples)
                / max(len(samples), 1)
            ),
        }
        critic.save(update_dir / "critic.json")
        update_result = train_pappo_ppo_update(
            model_name=args.model,
            samples=samples,
            output_dir=update_dir / "adapter",
            input_adapter_path=adapter_path,
            max_length=args.max_length,
            learning_rate=args.learning_rate,
            ppo_epochs=args.ppo_epochs,
            normalize_advantages_flag=not args.no_normalize_advantages,
            clip_epsilon=args.clip_epsilon,
            kl_beta=args.kl_beta,
            local_files_only=args.local_files_only,
        )
        metrics = {**critic_metrics, **asdict(update_result)}
        adapter_path = update_dir / "adapter"
        eval_rollouts = _rollout(
            backend_name=args.backend,
            model=args.model,
            adapter_path=adapter_path,
            manifest=args.manifest,
            start=args.train_limit,
            limit=args.eval_limit,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            local_files_only=args.local_files_only,
            seed=args.seed,
        )
        eval_metrics = evaluate_trajectories(eval_rollouts)
        _write_trajectories(
            args.output_dir / "eval" / f"update_{update_index:03d}_rollouts.jsonl",
            eval_rollouts,
        )
        _write_json(
            args.output_dir / "eval" / f"update_{update_index:03d}_metrics.json",
            eval_metrics,
        )
        metrics["eval_metrics"] = eval_metrics
        _write_json(update_dir / "metrics.json", metrics)
        update_reports.append(metrics)

    report = {
        "status": "pappo_ppo_completed",
        "updates": args.updates,
        "seed": args.seed,
        "train_tasks": args.train_limit,
        "eval_tasks": args.eval_limit,
        "num_rollouts_per_task": args.num_rollouts_per_task,
        "local_prior": args.local_prior,
        "normalize_advantages": not args.no_normalize_advantages,
        "base_metrics": base_metrics,
        "update_reports": update_reports,
    }
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
