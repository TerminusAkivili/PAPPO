"""Run a staged PAPPO-PPO experiment on SWE-bench Lite trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.ppo_rollout import (  # noqa: E402
    PPO_ACTION_SCOPES,
    PPO_REWARD_MODES,
    build_ppo_samples_from_trajectory,
)
from pappo.ppo_training import (  # noqa: E402
    apply_advantage_prior,
    fill_samples_with_model_logprobs,
    train_pappo_ppo_update,
)
from pappo.swe_bench_lite import (  # noqa: E402
    SWEBenchLiteLLMBackend,
    evaluate_patch_task,
    load_swe_bench_lite_tasks,
    write_trajectories,
)
from pappo.turn_critic import GroupMeanTurnCritic, MeanTurnCritic, RLOOTurnCritic  # noqa: E402


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_samples(path: Path, samples) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(asdict(sample), sort_keys=True) + "\n")


def _metric_summary(trajectories) -> dict[str, float]:
    return {
        "success_rate": (
            sum(float(trajectory.final_reward) for trajectory in trajectories)
            / len(trajectories)
            if trajectories
            else 0.0
        ),
        "trajectories": len(trajectories),
    }


def _patch_rollouts(args, *, limit: int, offset: int, update_index: int):
    tasks = load_swe_bench_lite_tasks(args.manifest, limit=offset + limit)[offset : offset + limit]
    trajectories = []
    for rollout_index in range(args.num_rollouts_per_task):
        for task in tasks:
            trajectories.append(
                evaluate_patch_task(
                    task,
                    work_root=args.output_dir / f"patch_work_update_{update_index:03d}_rollout_{rollout_index}",
                    timeout_seconds=args.timeout_seconds,
                )
            )
    return trajectories


def _eval_rollouts(args, adapter_path: Path | None):
    if args.eval_limit <= 0 or args.model == "tiny-local":
        return []
    tasks = load_swe_bench_lite_tasks(args.manifest, limit=args.train_limit + args.eval_limit)[
        args.train_limit : args.train_limit + args.eval_limit
    ]
    backend = SWEBenchLiteLLMBackend(
        model_path=args.model,
        adapter_path=adapter_path,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        local_files_only=args.local_files_only,
    )
    return [
        backend.repair(
            task,
            work_root=args.output_dir / "eval_work",
            timeout_seconds=args.timeout_seconds,
        )
        for task in tasks
    ]


def _fit_critic(samples, args):
    if args.critic_mode == "auto":
        critic_mode = "group_mean" if args.num_rollouts_per_task > 1 else "tool_mean"
    else:
        critic_mode = args.critic_mode
    if critic_mode == "rloo":
        critic = RLOOTurnCritic()
        sample_ids = [
            f"{sample.trajectory_id}:{sample.turn_id}:{index}"
            for index, sample in enumerate(samples)
        ]
        critic.fit(
            sample_ids=sample_ids,
            group_keys=[sample.trajectory_id for sample in samples],
            tool_names=[sample.tool_name for sample in samples],
            targets=[sample.target for sample in samples],
        )
        samples = [
            replace(
                sample,
                value=critic.predict(sample_id, sample.trajectory_id, sample.tool_name),
            )
            for sample_id, sample in zip(sample_ids, samples, strict=True)
        ]
        return samples, critic, "rloo"
    if critic_mode == "group_mean":
        critic = GroupMeanTurnCritic()
        critic.fit(
            group_keys=[sample.trajectory_id for sample in samples],
            tool_names=[sample.tool_name for sample in samples],
            targets=[sample.target for sample in samples],
        )
        return [
            replace(sample, value=critic.predict(sample.trajectory_id, sample.tool_name))
            for sample in samples
        ], critic, "group_mean"
    critic = MeanTurnCritic()
    critic.fit(
        tool_names=[sample.tool_name for sample in samples],
        targets=[sample.target for sample in samples],
    )
    return [
        replace(sample, value=critic.predict(sample.tool_name))
        for sample in samples
    ], critic, "tool_mean"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=70)
    parser.add_argument("--eval-limit", type=int, default=30)
    parser.add_argument("--updates", type=int, default=1)
    parser.add_argument("--num-rollouts-per-task", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--local-prior", type=float, default=0.25)
    parser.add_argument("--kl-beta", type=float, default=0.01)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--no-normalize-advantages", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--critic-mode",
        choices=["auto", "tool_mean", "group_mean", "rloo"],
        default="auto",
    )
    parser.add_argument(
        "--reward-mode",
        choices=list(PPO_REWARD_MODES),
        default="pappo_turn_local",
    )
    parser.add_argument(
        "--action-scope",
        choices=list(PPO_ACTION_SCOPES),
        default="edit",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_eval = _eval_rollouts(args, None)
    write_trajectories(args.output_dir / "eval" / "base_rollouts.jsonl", base_eval)
    _write_json(args.output_dir / "eval" / "base_metrics.json", _metric_summary(base_eval))

    adapter_path = None
    update_reports = []
    for update_index in range(args.updates):
        update_dir = args.output_dir / f"update_{update_index:03d}"
        train_rollouts = _patch_rollouts(
            args,
            limit=args.train_limit,
            offset=0,
            update_index=update_index,
        )
        write_trajectories(update_dir / "rollouts.jsonl", train_rollouts)
        samples = [
            sample
            for trajectory in train_rollouts
            for sample in build_ppo_samples_from_trajectory(
                trajectory,
                reward_mode=args.reward_mode,
                action_scope=args.action_scope,
            )
        ]
        samples, critic, critic_type = _fit_critic(samples, args)
        samples = apply_advantage_prior(samples, prior=args.local_prior)
        samples = fill_samples_with_model_logprobs(
            model_name=args.model,
            samples=samples,
            old_adapter_path=adapter_path,
            max_length=args.max_length,
            local_files_only=args.local_files_only,
        )
        _write_samples(update_dir / "ppo_samples.jsonl", samples)
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
        adapter_path = update_dir / "adapter"
        eval_rollouts = _eval_rollouts(args, adapter_path)
        write_trajectories(
            args.output_dir / "eval" / f"update_{update_index:03d}_rollouts.jsonl",
            eval_rollouts,
        )
        eval_metrics = _metric_summary(eval_rollouts)
        metrics = {
            "critic_type": critic_type,
            "critic_mode": args.critic_mode,
            "reward_mode": args.reward_mode,
            "action_scope": args.action_scope,
            "samples": len(samples),
            "eval_metrics": eval_metrics,
            **asdict(update_result),
        }
        _write_json(update_dir / "metrics.json", metrics)
        update_reports.append(metrics)

    report = {
        "status": "swe_bench_pappo_ppo_completed",
        "benchmark": "swe_bench_lite",
        "updates": args.updates,
        "seed": args.seed,
        "train_tasks": args.train_limit,
        "eval_tasks": args.eval_limit,
        "num_rollouts_per_task": args.num_rollouts_per_task,
        "critic_mode": args.critic_mode,
        "reward_mode": args.reward_mode,
        "action_scope": args.action_scope,
        "local_prior": args.local_prior,
        "normalize_advantages": not args.no_normalize_advantages,
        "update_reports": update_reports,
    }
    _write_json(args.output_dir / "report.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
