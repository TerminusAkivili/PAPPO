"""Audit the Qwen-only PAPPO top-paper experiment plan."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.top_paper import DEFAULT_TOP_PAPER_CONFIG, build_experiment_plan  # noqa: E402


def _status_counts(jobs) -> dict[str, int]:
    counts = Counter(job.status for job in jobs)
    return {
        "runnable": counts.get("planned", 0),
        "needs_runner": counts.get("needs_runner", 0),
        "total": len(jobs),
    }


def _needs_runner_summary(jobs) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for job in jobs:
        grouped[(job.benchmark, job.method)].append(int(job.seed))
    return [
        {
            "benchmark": benchmark,
            "method": method,
            "seeds": sorted(seeds),
        }
        for (benchmark, method), seeds in sorted(grouped.items())
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    plan = build_experiment_plan(DEFAULT_TOP_PAPER_CONFIG)
    by_benchmark: dict[str, list] = defaultdict(list)
    by_method: dict[str, list] = defaultdict(list)
    for job in plan.jobs:
        by_benchmark[job.benchmark].append(job)
        by_method[job.method].append(job)

    runnable = [job for job in plan.jobs if job.command]
    needs_runner = [job for job in plan.jobs if job.status == "needs_runner"]
    payload = {
        "status": "top_paper_audit_completed",
        "models": plan.models,
        "seeds": plan.seeds,
        "job_count": len(plan.jobs),
        "runnable_job_count": len(runnable),
        "needs_runner_count": len(needs_runner),
        "setup_command_count": len(plan.setup_commands),
        "by_benchmark": {
            name: _status_counts(jobs)
            for name, jobs in sorted(by_benchmark.items())
        },
        "by_method": {
            name: _status_counts(jobs)
            for name, jobs in sorted(by_method.items())
        },
        "gaps": plan.benchmark_gaps,
        "needs_runner_jobs": _needs_runner_summary(needs_runner),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
