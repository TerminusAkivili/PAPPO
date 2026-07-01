"""Train and optionally evaluate LoRA baselines on SWE-bench Lite trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.lora_training import LORA_METHODS, train_lora_adapter  # noqa: E402
from pappo.swe_bench_lite import (  # noqa: E402
    SWEBenchLiteLLMBackend,
    evaluate_patch_task,
    load_swe_bench_lite_tasks,
    write_trajectories,
)
from pappo.trajectory import trajectory_from_mapping  # noqa: E402


def _load_trajectories(path: Path):
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


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _train_rollouts(args):
    train_path = args.output_dir / "train_rollouts.jsonl"
    if args.reuse_train_rollouts and train_path.exists():
        return _load_trajectories(train_path)
    tasks = load_swe_bench_lite_tasks(args.manifest, limit=args.train_limit)
    trajectories = [
        evaluate_patch_task(
            task,
            work_root=args.output_dir / "train_work",
            timeout_seconds=args.timeout_seconds,
        )
        for task in tasks
    ]
    write_trajectories(train_path, trajectories)
    return trajectories


def _eval_adapter(args, adapter_dir: Path):
    if args.eval_limit <= 0:
        return None
    tasks = load_swe_bench_lite_tasks(args.manifest, limit=args.train_limit + args.eval_limit)[
        args.train_limit : args.train_limit + args.eval_limit
    ]
    backend = SWEBenchLiteLLMBackend(
        model_path=args.model,
        adapter_path=adapter_dir,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        local_files_only=args.local_files_only,
    )
    trajectories = [
        backend.repair(
            task,
            work_root=args.output_dir / "eval_work",
            timeout_seconds=args.timeout_seconds,
        )
        for task in tasks
    ]
    eval_path = args.output_dir / args.method / "eval_rollouts.jsonl"
    write_trajectories(eval_path, trajectories)
    return {
        "eval_rollouts": str(eval_path),
        "eval_trajectories": len(trajectories),
        "success_rate": (
            sum(float(trajectory.final_reward) for trajectory in trajectories)
            / len(trajectories)
            if trajectories
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=70)
    parser.add_argument("--eval-limit", type=int, default=30)
    parser.add_argument("--method", choices=list(LORA_METHODS), default="pappo_turn_v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--reuse-train-rollouts", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_trajectories = _train_rollouts(args)
    adapter_dir = args.output_dir / args.method / "adapter"
    train_result = train_lora_adapter(
        model_name=args.model,
        trajectories=train_trajectories,
        method=args.method,
        output_dir=adapter_dir,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        local_files_only=args.local_files_only,
    )
    eval_result = None if args.model == "tiny-local" else _eval_adapter(args, adapter_dir)
    payload = {
        "status": "swe_bench_lora_completed",
        "benchmark": "swe_bench_lite",
        "model": args.model,
        "method": args.method,
        "train_rollouts_count": len(train_trajectories),
        "train_report": asdict(train_result),
        "eval": eval_result,
    }
    _write_json(args.output_dir / "report.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
