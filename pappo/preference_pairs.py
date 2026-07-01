"""Preference-pair construction for PAPPO DPO/IPO-style baselines."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from pappo.trajectory import AgentTrajectory, split_tool_call_turns, trajectory_from_mapping
from pappo.trl_adapter import turn_response_text


@dataclass(frozen=True)
class PreferencePair:
    """One chosen/rejected response pair for preference training."""

    pair_id: str
    group_key: str
    prompt: str
    chosen: str
    rejected: str
    chosen_reward: float
    rejected_reward: float
    reward_delta: float
    chosen_trajectory_id: str
    rejected_trajectory_id: str


def trajectory_preference_group(trajectory: AgentTrajectory) -> str:
    """Return the most stable task-level grouping key available."""

    task_id = str(trajectory.metadata.get("task_id", ""))
    if task_id:
        return task_id
    repo_dir = str(trajectory.metadata.get("repo_dir", ""))
    if repo_dir:
        marker = "/repo"
        return repo_dir.split(marker, 1)[0] if marker in repo_dir else repo_dir
    trajectory_id = trajectory.trajectory_id
    if trajectory_id.endswith("-failure"):
        return trajectory_id.removesuffix("-failure")
    return trajectory_id


def _trajectory_prompt_and_response(trajectory: AgentTrajectory) -> tuple[str, str] | None:
    turns = split_tool_call_turns(trajectory)
    if not turns:
        return None
    prompt = turns[0].prompt or "<task>"
    response = "\n".join(turn_response_text(turn) for turn in turns)
    return prompt, response


def build_preference_pairs(
    trajectories: list[AgentTrajectory],
    *,
    limit: int | None = None,
) -> list[PreferencePair]:
    """Build chosen/rejected pairs from grouped trajectories."""

    grouped: dict[str, list[AgentTrajectory]] = {}
    for trajectory in trajectories:
        grouped.setdefault(trajectory_preference_group(trajectory), []).append(trajectory)

    pairs: list[PreferencePair] = []
    for group_key in sorted(grouped):
        group = sorted(
            grouped[group_key],
            key=lambda trajectory: (trajectory.final_reward, trajectory.trajectory_id),
        )
        rejected_candidates = [item for item in group if float(item.final_reward) < 1.0]
        chosen_candidates = [item for item in group if float(item.final_reward) > 0.0]
        if not rejected_candidates or not chosen_candidates:
            continue
        rejected = rejected_candidates[0]
        chosen = chosen_candidates[-1]
        if float(chosen.final_reward) <= float(rejected.final_reward):
            continue
        chosen_text = _trajectory_prompt_and_response(chosen)
        rejected_text = _trajectory_prompt_and_response(rejected)
        if chosen_text is None or rejected_text is None:
            continue
        prompt, chosen_response = chosen_text
        _rejected_prompt, rejected_response = rejected_text
        pairs.append(
            PreferencePair(
                pair_id=f"{group_key}::chosen::{chosen.trajectory_id}::rejected::{rejected.trajectory_id}",
                group_key=group_key,
                prompt=prompt,
                chosen=chosen_response,
                rejected=rejected_response,
                chosen_reward=float(chosen.final_reward),
                rejected_reward=float(rejected.final_reward),
                reward_delta=float(chosen.final_reward - rejected.final_reward),
                chosen_trajectory_id=chosen.trajectory_id,
                rejected_trajectory_id=rejected.trajectory_id,
            )
        )
        if limit is not None and len(pairs) >= limit:
            return pairs
    return pairs


def load_trajectories(path: Path) -> list[AgentTrajectory]:
    """Load trajectory JSONL."""

    trajectories: list[AgentTrajectory] = []
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


def write_preference_pairs(path: Path, pairs: list[PreferencePair]) -> None:
    """Write preference pairs as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(asdict(pair), sort_keys=True) + "\n")


def load_preference_pairs(path: Path) -> list[PreferencePair]:
    """Load preference pair JSONL."""

    pairs: list[PreferencePair] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                pairs.append(PreferencePair(**json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"failed to parse {path}:{line_number}") from exc
    return pairs


def preference_pair_summary(pairs: list[PreferencePair]) -> dict[str, float]:
    """Return summary metrics for generated pairs."""

    return {
        "pairs": len(pairs),
        "mean_reward_delta": float(mean(pair.reward_delta for pair in pairs)) if pairs else 0.0,
        "mean_chosen_reward": float(mean(pair.chosen_reward for pair in pairs)) if pairs else 0.0,
        "mean_rejected_reward": float(mean(pair.rejected_reward for pair in pairs)) if pairs else 0.0,
    }
