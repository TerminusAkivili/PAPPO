"""Build DPO/IPO-style preference pairs from PAPPO trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.preference_pairs import (  # noqa: E402
    build_preference_pairs,
    load_trajectories,
    preference_pair_summary,
    write_preference_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    trajectories = load_trajectories(args.trajectories)
    pairs = build_preference_pairs(trajectories, limit=args.limit)
    write_preference_pairs(args.output, pairs)
    payload = {
        "status": "preference_pairs_completed",
        "trajectories": str(args.trajectories),
        "output": str(args.output),
        **preference_pair_summary(pairs),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
