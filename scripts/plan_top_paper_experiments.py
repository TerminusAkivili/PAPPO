"""Generate the Qwen-only PAPPO top-paper experiment plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.top_paper import DEFAULT_TOP_PAPER_CONFIG, build_experiment_plan, write_plan  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/top_paper_qwen/experiment_plan.json"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only create the plan; no training jobs are executed by this script.",
    )
    args = parser.parse_args()

    plan = build_experiment_plan(DEFAULT_TOP_PAPER_CONFIG)
    write_plan(args.output, plan)
    payload = plan.to_dict()
    payload["output"] = str(args.output)
    payload["dry_run"] = bool(args.dry_run)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
