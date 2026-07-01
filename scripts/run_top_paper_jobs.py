"""Filter and optionally execute PAPPO top-paper experiment jobs."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.top_paper import DEFAULT_TOP_PAPER_CONFIG, ExperimentJob, build_experiment_plan  # noqa: E402


def _matches(values: list[str] | None, candidate: str) -> bool:
    return not values or candidate in values


def _selected_jobs(args) -> list[ExperimentJob]:
    plan = build_experiment_plan(DEFAULT_TOP_PAPER_CONFIG)
    jobs = [
        job
        for job in plan.jobs
        if job.command
        and _matches(args.benchmark, job.benchmark)
        and _matches(args.method, job.method)
        and (args.seed is None or job.seed in args.seed)
    ]
    return jobs[: args.limit] if args.limit is not None else jobs


def _shell_line(command: tuple[str, ...]) -> str:
    return " ".join(shlex.quote(item) for item in command)


def _write_shell(path: Path, jobs: list[ExperimentJob]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    lines.extend(_shell_line(job.command or ()) for job in jobs)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="append")
    parser.add_argument("--method", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-shell", type=Path)
    args = parser.parse_args()

    jobs = _selected_jobs(args)
    if args.write_shell is not None:
        _write_shell(args.write_shell, jobs)

    executed: list[dict[str, object]] = []
    if not args.dry_run:
        for job in jobs:
            completed = subprocess.run(list(job.command or ()), check=False)
            executed.append(
                {
                    "job_id": job.job_id,
                    "returncode": completed.returncode,
                }
            )
            if completed.returncode != 0:
                break

    payload = {
        "status": "top_paper_jobs_planned" if args.dry_run else "top_paper_jobs_executed",
        "dry_run": bool(args.dry_run),
        "selected_job_count": len(jobs),
        "write_shell": str(args.write_shell) if args.write_shell else None,
        "jobs": [
            {
                "job_id": job.job_id,
                "benchmark": job.benchmark,
                "method": job.method,
                "seed": job.seed,
                "command": list(job.command or ()),
                "output_dir": job.output_dir,
            }
            for job in jobs
        ],
        "executed": executed,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if executed and executed[-1]["returncode"] != 0:
        raise SystemExit(int(executed[-1]["returncode"]))


if __name__ == "__main__":
    main()
