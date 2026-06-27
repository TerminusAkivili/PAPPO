"""Generate executable RealRepoFix tasks and PAPPO trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.realrepofix import (  # noqa: E402
    apply_expected_fix,
    export_realrepofix_mixed_trajectories,
    export_realrepofix_trajectories,
    generate_realrepofix_tasks,
    run_pytest_reward,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/realrepofix_100"))
    parser.add_argument("--num-tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/realrepofix_100_manifest.jsonl"),
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=Path("data/realrepofix_100_trajectories.jsonl"),
    )
    parser.add_argument(
        "--mixed-trajectories",
        action="store_true",
        help="Export one success and one failure trajectory per task.",
    )
    args = parser.parse_args()

    tasks = generate_realrepofix_tasks(args.root, args.num_tasks, args.seed)
    manifest_rows = []
    for task in tasks:
        before = run_pytest_reward(task)
        apply_expected_fix(task)
        after = run_pytest_reward(task)
        # Restore buggy source so trajectory export records the real repair.
        (task.repo_dir / task.source_file).write_text(
            task.buggy_source,
            encoding="utf-8",
        )
        manifest_rows.append(
            {
                "task_id": task.task_id,
                "repo_dir": str(task.repo_dir),
                "issue": task.issue,
                "source_file": str(task.source_file),
                "test_file": str(task.test_file),
                "test_command": list(task.test_command),
                "template": task.template,
                "pytest_before_reward": before.reward,
                "pytest_before_returncode": before.returncode,
                "pytest_after_reward": after.reward,
                "pytest_after_returncode": after.returncode,
            }
        )

    trajectories = (
        export_realrepofix_mixed_trajectories(tasks)
        if args.mixed_trajectories
        else export_realrepofix_trajectories(tasks)
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.trajectories.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with args.trajectories.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(asdict(trajectory), sort_keys=True) + "\n")

    print(
        f"wrote {len(tasks)} RealRepoFix tasks to {args.root}; "
        f"manifest={args.manifest}; trajectories={args.trajectories}"
    )


if __name__ == "__main__":
    main()
