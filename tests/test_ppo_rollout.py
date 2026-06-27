from __future__ import annotations

from pappo.ppo_rollout import build_ppo_samples_from_trajectory
from pappo.trajectory import AgentEvent, AgentTrajectory, MESSAGE, TOOL_CALL, TOOL_RESULT


def test_ppo_sample_uses_raw_generated_edit_not_tool_result_wrapper() -> None:
    trajectory = AgentTrajectory(
        trajectory_id="t1",
        final_reward=1.0,
        metadata={"task_id": "task-1"},
        events=(
            AgentEvent(kind=MESSAGE, content="Fix the bug."),
            AgentEvent(kind=TOOL_CALL, tool_name="edit", content="replace file"),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="edit",
                content="parsed patch text",
                metadata={
                    "patch_applied": True,
                    "raw_generated_text": "```python\ndef fixed():\n    return True\n```",
                },
            ),
            AgentEvent(kind=TOOL_CALL, tool_name="run_test", content="pytest"),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="run_test",
                content="passed",
                metadata={"pytest_reward": 1.0},
            ),
        ),
    )

    samples = build_ppo_samples_from_trajectory(trajectory)

    assert len(samples) == 1
    assert samples[0].tool_name == "edit"
    assert samples[0].action_text.startswith("```python")
    assert "<tool_result>" not in samples[0].action_text


def test_ppo_sample_uses_raw_generation_prompt_when_available() -> None:
    trajectory = AgentTrajectory(
        trajectory_id="t1",
        final_reward=1.0,
        metadata={"task_id": "task-1"},
        events=(
            AgentEvent(kind=MESSAGE, content="Trajectory wrapper prompt."),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="edit",
                content="replace file",
                metadata={"raw_prompt_text": "<chat>actual generation prompt\n"},
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="edit",
                content="parsed patch text",
                metadata={
                    "patch_applied": True,
                    "raw_generated_text": "def fixed(): return True",
                },
            ),
        ),
    )

    samples = build_ppo_samples_from_trajectory(trajectory)

    assert len(samples) == 1
    assert samples[0].prompt == "<chat>actual generation prompt\n"


def test_ppo_sample_penalizes_applied_edit_when_final_reward_fails() -> None:
    trajectory = AgentTrajectory(
        trajectory_id="t1",
        final_reward=0.0,
        metadata={"task_id": "task-1"},
        events=(
            AgentEvent(kind=MESSAGE, content="Fix the bug."),
            AgentEvent(kind=TOOL_CALL, tool_name="edit", content="replace file"),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="edit",
                content="patch applied",
                metadata={
                    "patch_applied": True,
                    "raw_generated_text": "def broken(): return False",
                },
            ),
            AgentEvent(kind=TOOL_CALL, tool_name="run_test", content="pytest"),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="run_test",
                content="failed",
                metadata={"pytest_reward": 0.0},
            ),
        ),
    )

    samples = build_ppo_samples_from_trajectory(trajectory)

    assert len(samples) == 1
    assert samples[0].target < 0.0
