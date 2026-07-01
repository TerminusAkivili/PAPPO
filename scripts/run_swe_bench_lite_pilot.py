"""Run a SWE-bench Lite subset patch/test pilot and emit PAPPO trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.swe_bench_lite import (  # noqa: E402
    SWEBenchLiteLLMBackend,
    SWEBenchLiteReActBackend,
    SWEBenchLiteReflexionBackend,
    evaluate_patch_task,
    load_swe_bench_lite_tasks,
    write_trajectories,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument(
        "--backend",
        choices=["patch", "hf", "react_retry", "reflexion"],
        default="patch",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--work-root", type=Path, default=Path("data/swe_bench_lite_work"))
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout-seconds", type=int, default=30)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    tasks = load_swe_bench_lite_tasks(args.manifest, limit=args.limit)
    if args.backend == "patch":
        trajectories = [
            evaluate_patch_task(
                task,
                work_root=args.work_root,
                timeout_seconds=args.timeout_seconds,
            )
            for task in tasks
        ]
    else:
        backend_cls = SWEBenchLiteLLMBackend
        if args.backend == "react_retry":
            backend_cls = SWEBenchLiteReActBackend
        elif args.backend == "reflexion":
            backend_cls = SWEBenchLiteReflexionBackend
        backend = backend_cls(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            local_files_only=args.no_download,
        )
        trajectories = [
            backend.repair(
                task,
                work_root=args.work_root,
                timeout_seconds=args.timeout_seconds,
            )
            for task in tasks
        ]
    write_trajectories(args.trajectories, trajectories)
    payload = {
        "status": "swe_bench_lite_pilot_completed",
        "benchmark": "swe_bench_lite",
        "backend": args.backend,
        "model_path": args.model_path if args.backend != "patch" else None,
        "manifest": str(args.manifest),
        "trajectories": str(args.trajectories),
        "num_trajectories": len(trajectories),
        "success_rate": (
            sum(float(trajectory.final_reward) for trajectory in trajectories)
            / len(trajectories)
            if trajectories
            else 0.0
        ),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
