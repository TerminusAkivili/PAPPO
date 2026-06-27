"""Compare LoRA weighting rules on RealRepoFix agent trajectories."""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.llm_agent_pilot import HuggingFaceRepairBackend, ScriptedRepairBackend  # noqa: E402
from pappo.lora_training import LORA_METHODS, build_training_examples, train_lora_adapter  # noqa: E402
from pappo.trajectory import AgentTrajectory, split_tool_call_turns, trajectory_from_mapping  # noqa: E402
from scripts.run_llm_agent_pilot import _load_tasks  # noqa: E402


def load_jsonl_trajectories(path: Path) -> list[AgentTrajectory]:
    """Load serialized agent trajectories from JSONL."""

    trajectories = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                trajectories.append(trajectory_from_mapping(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"failed to parse {path}:{line_number}") from exc
    return trajectories


def _write_trajectories(path: Path, trajectories: list[AgentTrajectory]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(asdict(trajectory), sort_keys=True) + "\n")


def _task_slice(manifest: Path, start: int, limit: int, seed: int | None):
    tasks = _load_tasks(manifest, start + limit if seed is None else 10_000)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(tasks)
    return tasks[start : start + limit]


def _make_backend(
    backend_name: str,
    model: str,
    adapter_path: Path | None,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
):
    if backend_name == "scripted":
        return ScriptedRepairBackend()
    if backend_name != "hf":
        raise ValueError(f"unknown backend: {backend_name}")
    return HuggingFaceRepairBackend(
        model_path=model,
        adapter_path=adapter_path,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        local_files_only=local_files_only,
    )


def _set_rollout_seed(seed: int | None, offset: int) -> None:
    if seed is None:
        return
    rollout_seed = int(seed) * 1_000_003 + int(offset)
    random.seed(rollout_seed)
    try:
        import numpy as np

        np.random.seed(rollout_seed % (2**32 - 1))
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(rollout_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(rollout_seed)
    except Exception:
        pass
    try:
        from transformers import set_seed

        set_seed(rollout_seed)
    except Exception:
        pass


def _rollout(
    *,
    backend_name: str,
    model: str,
    adapter_path: Path | None,
    manifest: Path,
    start: int,
    limit: int,
    max_new_tokens: int,
    temperature: float,
    local_files_only: bool,
    seed: int | None = None,
) -> list[AgentTrajectory]:
    tasks = _task_slice(manifest, start, limit, seed)
    backend = _make_backend(
        backend_name,
        model,
        adapter_path,
        max_new_tokens,
        temperature,
        local_files_only,
    )
    try:
        trajectories: list[AgentTrajectory] = []
        for task_index, task in enumerate(tasks):
            _set_rollout_seed(seed, start + task_index)
            trajectories.append(backend.repair(task))
        return trajectories
    finally:
        del backend
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _rollout_grouped_train(args) -> list[AgentTrajectory]:
    tasks = _task_slice(args.manifest, args.train_start, args.train_limit, args.seed)
    backend = _make_backend(
        args.backend,
        args.model,
        None,
        args.max_new_tokens,
        args.temperature,
        args.local_files_only,
    )
    trajectories: list[AgentTrajectory] = []
    try:
        for task_index, task in enumerate(tasks):
            for rollout_index in range(args.num_rollouts_per_task):
                _set_rollout_seed(
                    args.seed,
                    (args.train_start + task_index) * args.num_rollouts_per_task
                    + rollout_index,
                )
                trajectories.append(backend.repair(task))
        return trajectories
    finally:
        del backend
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _metric_mean(values: list[float]) -> float:
    return float(mean(values)) if values else 0.0


def evaluate_trajectories(trajectories: list[AgentTrajectory]) -> dict[str, float]:
    """Compute comparison metrics over rollout trajectories."""

    costs: list[float] = []
    edit_successes: list[float] = []
    failed_edits: list[float] = []
    repeated_tests: list[float] = []
    search_usefulness: list[float] = []
    retry_rewards: list[float] = []

    for trajectory in trajectories:
        turns = split_tool_call_turns(trajectory)
        costs.append(float(sum(turn.cost for turn in turns)))
        run_test_count = sum(1 for turn in turns if turn.tool_name == "run_test")
        repeated_tests.append(float(run_test_count > 1))
        if trajectory.metadata.get("template") == "retry_permanent_errors":
            retry_rewards.append(float(trajectory.final_reward))

        for turn in turns:
            result_metadata = dict(turn.metadata.get("result_metadata", {}))
            if turn.tool_name == "edit":
                patch_applied = bool(result_metadata.get("patch_applied", False))
                edit_successes.append(float(patch_applied and trajectory.final_reward > 0.0))
                failed_edits.append(float(patch_applied and trajectory.final_reward <= 0.0))
            if turn.tool_name == "search":
                content = turn.tool_result
                search_usefulness.append(
                    float("tests/" in content and "assert " in content)
                )

    success_rate = _metric_mean([float(t.final_reward) for t in trajectories])
    return {
        "success_rate": success_rate,
        "avg_tool_cost": _metric_mean(costs),
        "edit_success_rate": _metric_mean(edit_successes),
        "failed_edit_rate": _metric_mean(failed_edits),
        "repeated_test_rate": _metric_mean(repeated_tests),
        "search_usefulness": _metric_mean(search_usefulness),
        "retry_task_success_rate": _metric_mean(retry_rewards),
    }


def _write_markdown_report(path: Path, report: dict) -> None:
    lines = [
        "# RealRepoFix LoRA Comparison",
        "",
        f"- Status: `{report['status']}`",
        f"- Backend: `{report['backend']}`",
        f"- Model: `{report['model']}`",
        f"- Train tasks: {report['train_tasks']}",
        f"- Eval tasks: {report['eval_tasks']}",
        "",
        "| Method | Success | Avg Cost | Edit Success | Failed Edit | Repeated Test | Search Useful | Retry Success |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in report["methods"]:
        metrics = report["metrics"][method]
        delta = report.get("delta_vs_base", {}).get(method, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    f"{metrics['success_rate']:.4f}",
                    f"{metrics['avg_tool_cost']:.2f}",
                    f"{metrics['edit_success_rate']:.4f}",
                    f"{metrics['failed_edit_rate']:.4f}",
                    f"{metrics['repeated_test_rate']:.4f}",
                    f"{metrics['search_usefulness']:.4f}",
                    f"{metrics['retry_task_success_rate']:.4f}",
                ]
            )
            + " |"
        )
    if "base_metrics" in report:
        lines.extend(
            [
                "",
                "## Delta vs Base",
                "",
                "| Method | Success Delta | Cost Delta | Edit Success Delta | Failed Edit Delta |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for method in report["methods"]:
            delta = report["delta_vs_base"][method]
            lines.append(
                "| "
                + " | ".join(
                    [
                        method,
                        f"{delta['success_rate']:+.4f}",
                        f"{delta['avg_tool_cost']:+.2f}",
                        f"{delta['edit_success_rate']:+.4f}",
                        f"{delta['failed_edit_rate']:+.4f}",
                    ]
                )
                + " |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _train_rollouts_path(output_dir: Path) -> Path:
    return output_dir / "train_rollouts.jsonl"


def _method_train_report_path(output_dir: Path, method: str) -> Path:
    return output_dir / method / "train_report.json"


def _method_eval_rollouts_path(output_dir: Path, method: str) -> Path:
    return output_dir / method / "eval_rollouts.jsonl"


def _method_metrics_path(output_dir: Path, method: str) -> Path:
    return output_dir / method / "metrics.json"


def _method_adapter_dir(output_dir: Path, method: str) -> Path:
    return output_dir / method / "adapter"


def _method_checkpoint_dir(output_dir: Path, method: str, after_trajectories: int) -> Path:
    return output_dir / method / "checkpoints" / f"after_{after_trajectories}"


def _base_eval_rollouts_path(output_dir: Path) -> Path:
    return output_dir / "base_eval_rollouts.jsonl"


def _base_metrics_path(output_dir: Path) -> Path:
    return output_dir / "base_metrics.json"


def _run_rollout_train(args) -> dict:
    train_path = _train_rollouts_path(args.output_dir)
    if args.reuse_train_rollouts and train_path.exists():
        train_trajectories = load_jsonl_trajectories(train_path)
    else:
        if args.num_rollouts_per_task > 1:
            train_trajectories = _rollout_grouped_train(args)
        else:
            train_trajectories = _rollout(
                backend_name=args.backend,
                model=args.model,
                adapter_path=None,
                manifest=args.manifest,
                start=args.train_start,
                limit=args.train_limit,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                local_files_only=args.local_files_only,
                seed=args.seed,
            )
        _write_trajectories(train_path, train_trajectories)
    return {
        "status": "rollout_train_completed",
        "seed": args.seed,
        "train_start": args.train_start,
        "eval_start": args.eval_start,
        "train_tasks": args.train_limit,
        "train_rollouts_count": len(train_trajectories),
        "train_rollouts": str(train_path),
    }


def _checkpoint_dirs_by_example(
    *,
    output_dir: Path,
    method: str,
    train_trajectories: list[AgentTrajectory],
    checkpoint_every: int,
) -> dict[int, Path]:
    if checkpoint_every <= 0:
        return {}
    checkpoint_dirs: dict[int, Path] = {}
    for after_count in range(checkpoint_every, len(train_trajectories) + 1, checkpoint_every):
        example_count = len(
            build_training_examples(train_trajectories[:after_count], method)
        )
        if example_count > 0:
            checkpoint_dirs[example_count] = _method_checkpoint_dir(
                output_dir, method, after_count
            )
    return checkpoint_dirs


def _run_train_adapter(args, method: str) -> dict:
    train_path = _train_rollouts_path(args.output_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"missing train rollouts: {train_path}")
    train_trajectories = load_jsonl_trajectories(train_path)
    adapter_dir = _method_adapter_dir(args.output_dir, method)
    checkpoint_dirs = _checkpoint_dirs_by_example(
        output_dir=args.output_dir,
        method=method,
        train_trajectories=train_trajectories,
        checkpoint_every=args.checkpoint_every,
    )
    train_result = train_lora_adapter(
        model_name=args.model,
        trajectories=train_trajectories,
        method=method,
        output_dir=adapter_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        local_files_only=args.local_files_only,
        checkpoint_dirs_by_example=checkpoint_dirs,
    )
    payload = asdict(train_result)
    _write_json(_method_train_report_path(args.output_dir, method), payload)
    return payload


def _run_eval_adapter(args, method: str) -> dict:
    adapter_dir = _method_adapter_dir(args.output_dir, method)
    if args.backend != "scripted" and not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"missing adapter for {method}: {adapter_dir}")
    eval_backend_adapter = None if args.backend == "scripted" else adapter_dir
    eval_trajectories = _rollout(
        backend_name=args.backend,
        model=args.model,
        adapter_path=eval_backend_adapter,
        manifest=args.manifest,
        start=args.eval_start,
        limit=args.eval_limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )
    _write_trajectories(_method_eval_rollouts_path(args.output_dir, method), eval_trajectories)
    metrics = evaluate_trajectories(eval_trajectories)
    _write_json(_method_metrics_path(args.output_dir, method), metrics)
    return metrics


def _run_eval_base(args) -> dict:
    eval_trajectories = _rollout(
        backend_name=args.backend,
        model=args.model,
        adapter_path=None,
        manifest=args.manifest,
        start=args.eval_start,
        limit=args.eval_limit,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        local_files_only=args.local_files_only,
        seed=args.seed,
    )
    _write_trajectories(_base_eval_rollouts_path(args.output_dir), eval_trajectories)
    metrics = evaluate_trajectories(eval_trajectories)
    _write_json(_base_metrics_path(args.output_dir), metrics)
    return metrics


def _metric_delta(metrics: dict[str, float], base_metrics: dict[str, float]) -> dict[str, float]:
    return {
        key: float(metrics.get(key, 0.0) - base_metrics.get(key, 0.0))
        for key in sorted(set(metrics) | set(base_metrics))
    }


def _run_summarize(args, methods: list[str]) -> dict:
    metrics: dict[str, dict[str, float]] = {}
    train_reports: dict[str, dict] = {}
    adapter_dirs: dict[str, str] = {}
    for method in methods:
        metrics_path = _method_metrics_path(args.output_dir, method)
        train_report_path = _method_train_report_path(args.output_dir, method)
        if not metrics_path.exists():
            raise FileNotFoundError(f"missing metrics for {method}: {metrics_path}")
        if not train_report_path.exists():
            raise FileNotFoundError(f"missing train report for {method}: {train_report_path}")
        metrics[method] = _read_json(metrics_path)
        train_reports[method] = _read_json(train_report_path)
        adapter_dirs[method] = str(_method_adapter_dir(args.output_dir, method))

    base_metrics = None
    base_metrics_path = _base_metrics_path(args.output_dir)
    if base_metrics_path.exists():
        base_metrics = _read_json(base_metrics_path)

    report = {
        "status": "realrepo_lora_comparison_completed",
        "backend": args.backend,
        "model": args.model,
        "seed": args.seed,
        "train_start": args.train_start,
        "eval_start": args.eval_start,
        "train_tasks": args.train_limit,
        "eval_tasks": args.eval_limit,
        "methods": methods,
        "metrics": metrics,
        "train_reports": train_reports,
        "adapter_dirs": adapter_dirs,
    }
    if base_metrics is not None:
        report["base_metrics"] = base_metrics
        report["delta_vs_base"] = {
            method: _metric_delta(method_metrics, base_metrics)
            for method, method_metrics in metrics.items()
        }
    _write_json(args.output_dir / "report.json", report)
    _write_markdown_report(args.output_dir / "report.md", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--backend", choices=["hf", "scripted"], default="hf")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=30)
    parser.add_argument("--eval-start", type=int)
    parser.add_argument("--eval-limit", type=int, default=30)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--methods", nargs="+", default=list(LORA_METHODS))
    parser.add_argument("--method", choices=list(LORA_METHODS))
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "rollout-train",
            "train-adapter",
            "eval-adapter",
            "eval-base",
            "summarize",
        ],
        default="all",
    )
    parser.add_argument("--reuse-train-rollouts", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--eval-base", action="store_true")
    parser.add_argument("--num-rollouts-per-task", type=int, default=1)
    args = parser.parse_args()
    if args.eval_start is None:
        args.eval_start = args.train_start + args.train_limit

    methods = [args.method] if args.method is not None else list(args.methods)
    unknown_methods = [method for method in methods if method not in LORA_METHODS]
    if unknown_methods:
        raise ValueError(f"unknown methods: {unknown_methods}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "rollout-train":
        result = _run_rollout_train(args)
    elif args.stage == "train-adapter":
        if args.method is None:
            raise ValueError("--stage train-adapter requires --method")
        result = _run_train_adapter(args, args.method)
    elif args.stage == "eval-adapter":
        if args.method is None:
            raise ValueError("--stage eval-adapter requires --method")
        result = _run_eval_adapter(args, args.method)
    elif args.stage == "eval-base":
        result = _run_eval_base(args)
    elif args.stage == "summarize":
        result = _run_summarize(args, methods)
    else:
        _run_rollout_train(args)
        if args.eval_base:
            _run_eval_base(args)
        for method in methods:
            _run_train_adapter(args, method)
            _run_eval_adapter(args, method)
        result = _run_summarize(args, methods)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
