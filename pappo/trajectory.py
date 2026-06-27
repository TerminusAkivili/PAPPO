"""Trajectory parsing for PAPPO-v1.

PAPPO-v1 treats a tool-call turn as the primary credit-assignment unit:

    context/reasoning -> tool_call -> tool_result

The parser below intentionally accepts a simple JSON-like event format so we can
reuse it for synthetic data, exported agent logs, and future TRL/OpenRLHF
adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
MESSAGE = "message"


@dataclass(frozen=True)
class AgentEvent:
    """One event in a coding-agent trajectory."""

    kind: str
    content: str = ""
    tool_name: str | None = None
    cost: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallTurn:
    """A trainable turn centered on one tool call."""

    turn_id: int
    tool_name: str
    prompt: str
    tool_call: str
    tool_result: str
    start_event: int
    end_event: int
    cost: float = 0.0
    final_reward: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Return the text span used by tokenization/collation."""

        pieces = [
            self.prompt,
            f"<tool_call name=\"{self.tool_name}\">\n{self.tool_call}\n</tool_call>",
            f"<tool_result>\n{self.tool_result}\n</tool_result>",
        ]
        return "\n".join(piece for piece in pieces if piece)


@dataclass(frozen=True)
class AgentTrajectory:
    """A full coding-agent rollout."""

    trajectory_id: str
    events: tuple[AgentEvent, ...]
    final_reward: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


def event_from_mapping(raw: Mapping[str, Any]) -> AgentEvent:
    """Parse one event from a JSON-like mapping."""

    return AgentEvent(
        kind=str(raw["kind"]),
        content=str(raw.get("content", "")),
        tool_name=(
            str(raw["tool_name"]) if raw.get("tool_name") is not None else None
        ),
        cost=float(raw.get("cost", 0.0)),
        metadata=dict(raw.get("metadata", {})),
    )


def trajectory_from_mapping(raw: Mapping[str, Any]) -> AgentTrajectory:
    """Parse a trajectory from a JSON-like mapping."""

    return AgentTrajectory(
        trajectory_id=str(raw["trajectory_id"]),
        events=tuple(event_from_mapping(event) for event in raw["events"]),
        final_reward=float(raw["final_reward"]),
        metadata=dict(raw.get("metadata", {})),
    )


def split_tool_call_turns(trajectory: AgentTrajectory) -> list[ToolCallTurn]:
    """Split a trajectory into tool-call turns.

    A turn starts at a `tool_call` event and ends at the following `tool_result`
    event. Message events between the previous result and the current call are
    used as the prompt/reasoning context for the turn.
    """

    turns: list[ToolCallTurn] = []
    context: list[str] = []
    index = 0

    while index < len(trajectory.events):
        event = trajectory.events[index]
        if event.kind == MESSAGE:
            if event.content:
                context.append(event.content)
            index += 1
            continue

        if event.kind != TOOL_CALL:
            index += 1
            continue

        result_index = index + 1
        while result_index < len(trajectory.events):
            result_event = trajectory.events[result_index]
            if result_event.kind == TOOL_RESULT:
                break
            result_index += 1
        else:
            raise ValueError(
                f"tool_call at event {index} has no following tool_result"
            )

        result_event = trajectory.events[result_index]
        tool_name = event.tool_name or result_event.tool_name or "unknown"
        turn = ToolCallTurn(
            turn_id=len(turns),
            tool_name=tool_name,
            prompt="\n".join(context),
            tool_call=event.content,
            tool_result=result_event.content,
            start_event=index,
            end_event=result_index,
            cost=event.cost + result_event.cost,
            final_reward=trajectory.final_reward,
            metadata={
                "trajectory_id": trajectory.trajectory_id,
                "call_metadata": dict(event.metadata),
                "result_metadata": dict(result_event.metadata),
            },
        )
        turns.append(turn)

        context = [turn.text]
        index = result_index + 1

    return turns
