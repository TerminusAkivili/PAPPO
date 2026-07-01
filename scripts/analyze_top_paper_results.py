"""Analyze PAPPO top-paper rollout artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.top_paper import (  # noqa: E402
    bootstrap_success_ci,
    failure_case_examples,
    failure_categories,
    load_trajectory_jsonl,
    mechanism_metrics,
    paired_win_loss,
    sign_test_p_value,
    task_rewards,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-rollouts", type=Path, required=True)
    parser.add_argument("--baseline-rollouts", type=Path)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate = load_trajectory_jsonl(args.candidate_rollouts)
    candidate_rewards = [float(trajectory.final_reward) for trajectory in candidate]
    payload = {
        "status": "top_paper_analysis_completed",
        "candidate_rollouts": str(args.candidate_rollouts),
        "candidate_count": len(candidate),
        "candidate_success_ci": bootstrap_success_ci(
            candidate_rewards,
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        ),
        "mechanism_metrics": mechanism_metrics(candidate),
        "failure_categories": failure_categories(candidate),
    }

    if args.baseline_rollouts is not None:
        baseline = load_trajectory_jsonl(args.baseline_rollouts)
        paired = paired_win_loss(
            baseline=task_rewards(baseline),
            candidate=task_rewards(candidate),
        )
        payload["baseline_rollouts"] = str(args.baseline_rollouts)
        payload["baseline_count"] = len(baseline)
        payload["baseline_success_ci"] = bootstrap_success_ci(
            [float(trajectory.final_reward) for trajectory in baseline],
            iterations=args.bootstrap_iterations,
            seed=args.seed,
        )
        payload["paired_win_loss"] = paired
        payload["paired_sign_test_p"] = sign_test_p_value(
            wins=int(paired["wins"]),
            losses=int(paired["losses"]),
        )
        payload["case_examples"] = failure_case_examples(
            baseline=baseline,
            candidate=candidate,
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
