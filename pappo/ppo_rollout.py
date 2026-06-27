"""Build PAPPO-PPO samples from agent trajectories."""

from __future__ import annotations

from pappo.lora_training import pappo_turn_v2_score
from pappo.ppo_training import PPOTurnSample
from pappo.trajectory import AgentTrajectory, split_tool_call_turns


def build_ppo_samples_from_trajectory(
    trajectory: AgentTrajectory,
) -> list[PPOTurnSample]:
    """Convert a trajectory into action-only edit samples for PPO."""

    samples: list[PPOTurnSample] = []
    for turn in split_tool_call_turns(trajectory):
        call_metadata = dict(turn.metadata.get("call_metadata", {}))
        result_metadata = dict(turn.metadata.get("result_metadata", {}))
        if turn.tool_name != "edit":
            continue
        raw_generated_text = result_metadata.get("raw_generated_text") or turn.tool_result
        prompt = call_metadata.get("raw_prompt_text") or turn.prompt or "<task>"
        target = pappo_turn_v2_score(trajectory, turn)
        samples.append(
            PPOTurnSample(
                trajectory_id=str(
                    turn.metadata.get("trajectory_id", trajectory.trajectory_id)
                ),
                turn_id=turn.turn_id,
                tool_name=turn.tool_name,
                prompt=str(prompt),
                action_text=str(raw_generated_text),
                target=float(target),
                value=0.0,
                old_logprobs=(),
                ref_logprobs=(),
                action_mask=(),
            )
        )
    return samples
