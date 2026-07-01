"""Build PAPPO-PPO samples from agent trajectories."""

from __future__ import annotations

from pappo.lora_training import pappo_turn_v2_score
from pappo.ppo_training import PPOTurnSample
from pappo.trajectory import AgentTrajectory, split_tool_call_turns


PPO_REWARD_MODES = (
    "pappo_turn_local",
    "trace",
    "token_broadcast",
    "terminal",
)
PPO_ACTION_SCOPES = ("edit", "all_tools")


def _turn_target(trajectory: AgentTrajectory, turn, reward_mode: str) -> float:
    if reward_mode not in PPO_REWARD_MODES:
        raise ValueError(f"unknown PPO reward mode: {reward_mode}")
    if reward_mode in {"trace", "token_broadcast", "terminal"}:
        return float(trajectory.final_reward)
    return float(pappo_turn_v2_score(trajectory, turn))


def _turn_action_text(turn) -> str:
    result_metadata = dict(turn.metadata.get("result_metadata", {}))
    if turn.tool_name == "edit":
        return str(result_metadata.get("raw_generated_text") or turn.tool_result)
    return str(turn.tool_call)


def build_ppo_samples_from_trajectory(
    trajectory: AgentTrajectory,
    *,
    reward_mode: str = "pappo_turn_local",
    action_scope: str = "edit",
) -> list[PPOTurnSample]:
    """Convert a trajectory into action-only turn samples for PPO."""

    if action_scope not in PPO_ACTION_SCOPES:
        raise ValueError(f"unknown PPO action scope: {action_scope}")
    samples: list[PPOTurnSample] = []
    for turn in split_tool_call_turns(trajectory):
        call_metadata = dict(turn.metadata.get("call_metadata", {}))
        if action_scope == "edit" and turn.tool_name != "edit":
            continue
        prompt = call_metadata.get("raw_prompt_text") or turn.prompt or "<task>"
        target = _turn_target(trajectory, turn, reward_mode)
        samples.append(
            PPOTurnSample(
                trajectory_id=str(
                    turn.metadata.get("trajectory_id", trajectory.trajectory_id)
                ),
                turn_id=turn.turn_id,
                tool_name=turn.tool_name,
                prompt=str(prompt),
                action_text=_turn_action_text(turn),
                target=float(target),
                value=0.0,
                old_logprobs=(),
                ref_logprobs=(),
                action_mask=(),
            )
        )
    return samples
