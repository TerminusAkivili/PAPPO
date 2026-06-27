"""Minimal TRL adapter for PAPPO-v1.

TRL already owns the PPO loop and value-head language model. PAPPO-v1 only
needs to map tool-call turns into the `PPOTrainer.step` interface:

    queries, responses, scores, response_masks
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from pappo.trajectory import ToolCallTurn


class TokenizerLike(Protocol):
    """Small tokenizer protocol required by the TRL adapter."""

    pad_token_id: int | None

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        """Tokenize text into input ids."""


@dataclass(frozen=True)
class TRLPPOBatch:
    """PAPPO-v1 batch ready to pass into `PPOTrainer.step`."""

    queries: list[torch.LongTensor]
    responses: list[torch.LongTensor]
    scores: list[torch.FloatTensor]
    response_masks: list[torch.LongTensor]


def turn_response_text(turn: ToolCallTurn) -> str:
    """Return the trainable response span for a tool-call turn."""

    return "\n".join(
        [
            f"<tool_call name=\"{turn.tool_name}\">",
            turn.tool_call,
            "</tool_call>",
            "<tool_result>",
            turn.tool_result,
            "</tool_result>",
        ]
    )


def _encode(
    tokenizer: TokenizerLike,
    text: str,
    max_length: int,
) -> torch.LongTensor:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )
    return torch.tensor(encoded["input_ids"], dtype=torch.long)


def build_trl_ppo_batch(
    tokenizer: TokenizerLike,
    turns: list[ToolCallTurn],
    scores: list[float],
    max_query_length: int = 1024,
    max_response_length: int = 1024,
) -> TRLPPOBatch:
    """Build a PAPPO-v1 batch for TRL PPO.

    `scores` can be final rewards, turn-level rewards, or normalized turn
    advantages depending on the experiment. The important PAPPO-v1 choice is
    that one score is attached to one tool-call turn.
    """

    if len(turns) != len(scores):
        raise ValueError("turns and scores must have the same length")

    queries: list[torch.LongTensor] = []
    responses: list[torch.LongTensor] = []
    score_tensors: list[torch.FloatTensor] = []
    response_masks: list[torch.LongTensor] = []

    for turn, score in zip(turns, scores, strict=True):
        response = _encode(
            tokenizer,
            turn_response_text(turn),
            max_length=max_response_length,
        )
        queries.append(
            _encode(tokenizer, turn.prompt or "<task>", max_length=max_query_length)
        )
        responses.append(response)
        score_tensors.append(torch.tensor(float(score), dtype=torch.float32))
        response_masks.append(torch.ones_like(response, dtype=torch.long))

    return TRLPPOBatch(
        queries=queries,
        responses=responses,
        scores=score_tensors,
        response_masks=response_masks,
    )
