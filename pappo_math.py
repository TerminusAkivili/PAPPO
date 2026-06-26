"""PAPPO research sketch.

PAPPO means Patch-Aware Proximal Policy Optimization.

This file is intentionally not a full training implementation. It records the
current mathematical direction for PAPPO in a Python-shaped form so the idea can
later grow into runnable code without losing the research structure.

Core thesis
-----------
For long-horizon coding agents, standard PPO is usually limited less by the
clipped surrogate itself and more by weak advantage estimation. A coding-agent
trajectory contains tool calls, patch edits, test runs, failed attempts,
state-changing repository updates, tool-call turns, explicit belief states, and
compacted sub-traces. PAPPO therefore makes the value model a first-class
object and uses structured advantages for success, patch progress, tool utility,
belief quality, future cost, compaction consistency, and failure risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PAPPOWeights:
    """Weights for the structured PAPPO advantage.

    The total advantage is:

        A_pappo =
            A_success
            + lambda_patch * A_patch
            + lambda_tool * A_tool
            + lambda_belief * A_belief
            + lambda_compact * A_compact
            - lambda_cost * A_cost
            - lambda_risk * A_risk

    Larger patch/tool weights make the policy more sensitive to useful
    intermediate progress. Larger cost/risk weights make it more conservative.
    """

    lambda_patch: float = 1.0
    lambda_tool: float = 1.0
    lambda_belief: float = 0.5
    lambda_compact: float = 0.5
    lambda_cost: float = 0.2
    lambda_risk: float = 0.5


@dataclass(frozen=True)
class StructuredAdvantage:
    """Per-step advantage components used by PAPPO."""

    success: float
    patch: float
    tool: float
    belief: float
    compact: float
    cost: float
    risk: float


@dataclass(frozen=True)
class StructuredValue:
    """Outputs of a strong value model for a coding-agent state.

    V_success:
        Expected future task success or final task return.

    V_patch:
        Expected quality of the current repository/patch state.

    V_tool:
        Expected marginal utility of taking a specific tool action from the
        current state.

    V_belief:
        Expected usefulness and faithfulness of the agent's explicit belief
        state. This measures whether the agent's current compact task model is
        likely to support future progress.

    V_cost:
        Expected remaining tool, token, test, or wall-clock cost.

    V_compact:
        Expected value preservation across compaction and sub-trace boundaries.
        This asks whether a compacted context still contains the state needed to
        continue solving the task.

    V_risk:
        Probability or expected severity of entering a bad trajectory region,
        such as reward hacking, repeated useless search, or destructive edits.
    """

    success: float
    patch: float
    tool: float
    belief: float
    compact: float
    cost: float
    risk: float


@dataclass(frozen=True)
class ToolCallTurn:
    """A natural critic step for agentic coding RL.

    A turn is coarser than a token but much smaller than a full trace:

        belief_before -> reasoning/action -> tool_call -> tool_result -> belief_after

    This is the main PAPPO credit-assignment unit. Token-level loss can still be
    used inside a turn, but the critic target should primarily explain whether
    the turn improved the agent's future repair probability.
    """

    tool: str
    start_token: int
    end_token: int
    cost: float
    hack_detected: bool = False


@dataclass(frozen=True)
class BeliefState:
    """Explicit compact state maintained by the coding agent.

    The belief block is a structured summary of what the agent currently thinks
    is true about the task, repository, failure mode, patch plan, and remaining
    uncertainty. It can make both actor and critic training easier than forcing
    them to reconstruct the task state from a long raw context.
    """

    task_summary: str
    relevant_files: tuple[str, ...]
    failure_hypothesis: str
    patch_plan: str
    uncertainty: str


@dataclass(frozen=True)
class SubTrace:
    """A compacted trainable unit from a long coding-agent rollout.

    GLM-5.2's long-horizon setup suggests that full rollouts should be split by
    compaction into unequal trainable sub-traces. PAPPO treats each sub-trace as
    a valid PPO training sequence while preserving the original rollout id for
    analysis and value consistency checks.
    """

    rollout_id: str
    start_step: int
    end_step: int
    token_count: int
    compacted: bool
    terminal: bool = False


class ValueModel(Protocol):
    """Protocol for a structured PAPPO critic."""

    def predict(self, state: object, tool: str | None = None) -> StructuredValue:
        """Return structured value estimates for a state and optional tool."""


def pappo_advantage(
    advantage: StructuredAdvantage,
    weights: PAPPOWeights = PAPPOWeights(),
) -> float:
    """Combine structured advantages into the scalar used by PPO.

    This scalar can replace the standard PPO advantage inside the clipped
    surrogate. The structure stays visible for logging and interpretability.
    """

    return (
        advantage.success
        + weights.lambda_patch * advantage.patch
        + weights.lambda_tool * advantage.tool
        + weights.lambda_belief * advantage.belief
        + weights.lambda_compact * advantage.compact
        - weights.lambda_cost * advantage.cost
        - weights.lambda_risk * advantage.risk
    )


def clipped_policy_objective(
    probability_ratio: float,
    advantage: float,
    epsilon: float = 0.2,
) -> float:
    """PPO clipped surrogate for one action.

    Standard PPO:

        L(theta) = min(
            r_t(theta) * A_t,
            clip(r_t(theta), 1 - eps, 1 + eps) * A_t
        )

    PAPPO uses the same trust-region style surrogate, but A_t is the structured
    PAPPO advantage rather than a single return-minus-value estimate.
    """

    clipped_ratio = min(max(probability_ratio, 1.0 - epsilon), 1.0 + epsilon)
    return min(probability_ratio * advantage, clipped_ratio * advantage)


def token_level_pappo_objective(
    probability_ratios: list[float],
    advantages: list[float],
    epsilon: float = 0.2,
) -> float:
    """Length-balanced PPO objective for compacted sub-traces.

    Long-horizon coding tasks produce sub-traces with very different lengths.
    Summing token losses would let long traces dominate the update. PAPPO uses
    a token-level objective averaged inside each trainable trace.
    """

    if len(probability_ratios) != len(advantages):
        raise ValueError("probability_ratios and advantages must have same length")
    if not probability_ratios:
        return 0.0

    token_losses = [
        clipped_policy_objective(ratio, advantage, epsilon)
        for ratio, advantage in zip(probability_ratios, advantages, strict=True)
    ]
    return sum(token_losses) / len(token_losses)


def turn_level_pappo_objective(
    turns: list[ToolCallTurn],
    token_probability_ratios: list[list[float]],
    turn_advantages: list[float],
    epsilon: float = 0.2,
) -> float:
    """PAPPO objective averaged at tool-call turn granularity.

    The key design choice from recent long-horizon agentic RL discussion is that
    the critic should usually not operate at pure token granularity. Token-level
    value can have poor signal-to-noise. Full-trace value is too coarse. A
    tool-call turn is a cheap, scalable middle unit: it contains an action, an
    external result, and enough semantic mass for incremental evaluation.

    For each turn, PAPPO applies one turn advantage to all policy tokens inside
    that turn, averages token losses within the turn, then averages across
    turns. This prevents long turns from dominating while preserving PPO's
    token-level log-prob machinery.
    """

    if not (
        len(turns) == len(token_probability_ratios) == len(turn_advantages)
    ):
        raise ValueError("turns, ratios, and advantages must have same length")
    if not turns:
        return 0.0

    turn_losses = []
    for ratios, advantage in zip(
        token_probability_ratios, turn_advantages, strict=True
    ):
        turn_losses.append(
            token_level_pappo_objective(
                probability_ratios=ratios,
                advantages=[advantage] * len(ratios),
                epsilon=epsilon,
            )
        )
    return sum(turn_losses) / len(turn_losses)


def tool_marginal_utility(
    value_before: StructuredValue,
    value_after: StructuredValue,
    tool_cost: float,
    cost_weight: float = 1.0,
) -> float:
    """Estimate whether a tool call was worth taking.

    Tool utility should not mean "use fewer tools." It should mean:

        did this tool buy enough future success probability or patch progress
        to justify its cost?

    A useful expensive test run can therefore receive positive utility, while a
    cheap but useless search can receive negative utility.
    """

    delta_success = value_after.success - value_before.success
    delta_patch = value_after.patch - value_before.patch
    delta_belief = value_after.belief - value_before.belief
    return delta_success + delta_patch + delta_belief - cost_weight * tool_cost


def belief_update_delta(
    value_before_belief: StructuredValue,
    value_after_belief: StructuredValue,
    expected_delta: float = 0.0,
) -> float:
    """Credit for maintaining or improving the explicit belief block.

    A good belief update should make the current task state easier for both the
    actor and critic to use. It may improve future success even before any tool
    call is made, which gives PAPPO a step-like supervision target for
    reasoning-only spans as well.
    """

    observed_delta = (
        value_after_belief.success
        - value_before_belief.success
        + value_after_belief.belief
        - value_before_belief.belief
        + value_after_belief.compact
        - value_before_belief.compact
    )
    return observed_delta - expected_delta


def patch_delta(
    value_before_patch: StructuredValue,
    value_after_patch: StructuredValue,
    expected_delta: float = 0.0,
) -> float:
    """Patch-level credit signal.

    Coding agents differ from pure text reasoning agents because edits mutate
    the repository state. PAPPO should reward a patch when it makes the future
    repair more likely, even if the final trajectory still fails.

    The patch advantage can be estimated as:

        A_patch = (V_success(after) - V_success(before)) - E[delta | state]

    Here we also include V_patch so partial repository improvements are visible.
    """

    observed_delta = (
        value_after_patch.success
        - value_before_patch.success
        + value_after_patch.patch
        - value_before_patch.patch
    )
    return observed_delta - expected_delta


def compaction_consistency(
    value_before_compaction: StructuredValue,
    value_after_compaction: StructuredValue,
) -> float:
    """Measure value preservation across context compaction.

    Compaction is useful only if the compressed state preserves the information
    needed for future repair. A large negative delta means the compaction likely
    dropped important context; a near-zero or positive delta means the sub-trace
    can be trained without destroying credit assignment.
    """

    return (
        value_after_compaction.success
        - value_before_compaction.success
        + value_after_compaction.patch
        - value_before_compaction.patch
        + value_after_compaction.belief
        - value_before_compaction.belief
        + value_after_compaction.compact
        - value_before_compaction.compact
    )


def hack_adjusted_advantage(
    advantage: StructuredAdvantage,
    hack_detected: bool,
    blocked_online: bool,
    penalty: float = 1.0,
) -> StructuredAdvantage:
    """Adjust training signal for online anti-hack handling.

    The important design choice is to avoid discarding the whole trajectory when
    a suspicious tool call is detected. Instead, the invalid action receives a
    risk penalty, the environment can return dummy information, and the rollout
    may continue so useful later recovery behavior remains trainable.
    """

    if not hack_detected:
        return advantage

    risk_penalty = penalty if blocked_online else penalty * 2.0
    return StructuredAdvantage(
        success=advantage.success,
        patch=advantage.patch,
        tool=advantage.tool,
        belief=advantage.belief,
        compact=advantage.compact,
        cost=advantage.cost,
        risk=advantage.risk + risk_penalty,
    )


def value_model_loss_terms() -> dict[str, str]:
    """Document the intended multi-task value-model training losses.

    A strong PAPPO critic should not only regress final returns. It should learn
    several future-facing signals that make coding trajectories explainable.
    """

    return {
        "L_success": "predict final task success or final return",
        "L_patch": "predict patch-state improvement after edits",
        "L_tool": "predict marginal value delta of tool calls",
        "L_belief": "predict usefulness and faithfulness of explicit belief updates",
        "L_compact": "predict whether compacted sub-traces preserve task state",
        "L_cost": "predict remaining tool, token, test, or wall-clock cost",
        "L_risk": "predict abnormal, hacked, looping, or destructive trajectories",
        "L_dist": "optional distributional value loss for multimodal outcomes",
    }


PAPPO_OBJECTIVE = r"""
Trajectory:
    tau = (s_0, a_0, o_1, s_1, ..., s_T)

Hybrid coding-agent action:
    a_t = (tool_t, args_t, emitted_tokens_t)

Primary credit-assignment unit:
    turn_k = (belief_before, reasoning/action, tool_call, tool_result, belief_after)

Task objective:
    J(theta) = E_{tau ~ pi_theta}[
        R_success(tau)
        + alpha * R_progress(tau)
        - beta * C_tool(tau)
        - gamma * R_risk(tau)
    ]

Structured PAPPO advantage:
    A_t^PAPPO =
        A_t^success
        + lambda_patch * A_t^patch
        + lambda_tool * A_t^tool
        + lambda_belief * A_t^belief
        + lambda_compact * A_t^compact
        - lambda_cost * A_t^cost
        - lambda_risk * A_t^risk

Turn-level PAPPO clipped surrogate:
    L_PAPPO(theta) = E_subtrace E_turn E_token_in_turn[
        min(
            r_i(theta) * A_turn^PAPPO,
            clip(r_i(theta), 1 - eps, 1 + eps) * A_turn^PAPPO
        )
    ]

where:
    r_i(theta) = pi_theta(token_i | context_i) / pi_old(token_i | context_i)

Compacted sub-trace training:
    long rollouts are split into variable-length trainable sub-traces.
    PAPPO learns from each sub-trace independently, while the structured value
    model tracks whether compaction preserved future task value.

Online anti-hack handling:
    suspicious tool calls are blocked or replaced with dummy observations,
    increasing A_risk for that action while preserving the rest of the rollout.

Belief block:
    the agent maintains an explicit belief state after observations and tool
    results. This gives the critic a compact target for evaluating process
    quality and gives the actor a stable state representation for long tasks.
"""


if __name__ == "__main__":
    example_advantage = StructuredAdvantage(
        success=0.8,
        patch=0.3,
        tool=0.2,
        belief=0.15,
        compact=0.1,
        cost=0.1,
        risk=0.05,
    )
    combined = pappo_advantage(example_advantage)
    objective = clipped_policy_objective(probability_ratio=1.1, advantage=combined)

    print("PAPPO combined advantage:", round(combined, 4))
    print("PAPPO clipped objective:", round(objective, 4))
    print(PAPPO_OBJECTIVE.strip())
