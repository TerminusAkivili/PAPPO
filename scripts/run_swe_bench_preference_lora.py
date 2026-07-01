"""Train a DPO/IPO-like preference LoRA baseline on SWE-bench Lite tasks."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.lora_training import TrainingExample, train_lora_from_examples  # noqa: E402
from pappo.preference_pairs import build_preference_pairs, write_preference_pairs  # noqa: E402
from pappo.swe_bench_lite import (  # noqa: E402
    SWEBenchLiteTask,
    evaluate_patch_task,
    load_swe_bench_lite_tasks,
    write_trajectories,
)


def _preference_examples(pairs) -> list[TrainingExample]:
    examples: list[TrainingExample] = []
    for pair in pairs:
        margin = max(float(pair.reward_delta), 1e-6)
        examples.append(
            TrainingExample(prompt=pair.prompt, response=pair.chosen, weight=margin)
        )
        examples.append(
            TrainingExample(prompt=pair.prompt, response=pair.rejected, weight=-margin)
        )
    return examples


def _mixed_rollouts(args):
    tasks = load_swe_bench_lite_tasks(args.manifest, limit=args.train_limit)
    trajectories = []
    for task in tasks:
        success = evaluate_patch_task(
            task,
            work_root=args.output_dir / "preference_work_success",
            timeout_seconds=args.timeout_seconds,
        )
        failure_task = SWEBenchLiteTask(
            task_id=task.task_id,
            repo_dir=task.repo_dir,
            problem_statement=task.problem_statement,
            patch="",
            test_command=task.test_command,
            source_file=task.source_file,
            base_commit=task.base_commit,
            repo=task.repo,
        )
        failure = evaluate_patch_task(
            failure_task,
            work_root=args.output_dir / "preference_work_failure",
            timeout_seconds=args.timeout_seconds,
        )
        trajectories.extend([success, failure])
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-limit", type=int, default=70)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = _mixed_rollouts(args)
    rollouts_path = args.output_dir / "preference_rollouts.jsonl"
    pairs_path = args.output_dir / "preference_pairs.jsonl"
    write_trajectories(rollouts_path, trajectories)
    pairs = build_preference_pairs(trajectories)
    write_preference_pairs(pairs_path, pairs)
    examples = _preference_examples(pairs)
    train_result = train_lora_from_examples(
        model_name=args.model,
        examples=examples,
        method="dpo_ipo_like",
        output_dir=args.output_dir / "adapter",
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        local_files_only=args.local_files_only,
    )
    payload = {
        "status": "swe_bench_preference_lora_completed",
        "benchmark": "swe_bench_lite",
        "method": "dpo_ipo_like",
        "train_rollouts_count": len(trajectories),
        "pairs": len(pairs),
        "examples": len(examples),
        "rollouts": str(rollouts_path),
        "pairs_path": str(pairs_path),
        "train_report": asdict(train_result),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
