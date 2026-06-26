# PAPPO

`PAPPO` is a research project on reinforcement learning for tool-using coding agents.

The core idea is simple: modern software engineering agents do not only generate code. They read repositories, search for symbols, edit files, run tests, inspect failures, and iterate over long trajectories. These behaviors are naturally sequential and budget-sensitive, which makes them a strong fit for policy optimization. PAPPO focuses on building a PPO-style algorithm specialized for this setting, together with the benchmarks and interpretability tools needed to study it rigorously.

## Project Overview

PAPPO stands for `Patch-Aware Proximal Policy Optimization`.

The project asks three connected questions:

1. How should PPO be adapted for multi-step coding agents that alternate between reading, searching, editing, and testing?
2. Which benchmark protocol best measures long-horizon coding-agent quality beyond final pass rate alone?
3. What internal signals explain why an agent decides to spend effort on a tool call, commit a patch, or stop iterating?

PAPPO is intentionally centered on the RL algorithm itself rather than a general-purpose runtime framework. The main deliverables are expected to be:

- a task-specific PPO variant for tool-using coding agents
- a reproducible benchmark recipe for long-horizon software engineering tasks
- interpretability analyses for agent behavior, cost trade-offs, and failure modes

## Research Architecture

The repository is organized around three research layers.

### 1. RL Algorithm

The algorithm layer studies a PPO variant for repository-level coding tasks.

Target properties:

- stable optimization on long trajectories
- better credit assignment across read, search, edit, and test actions
- support for tool-aware or patch-aware advantages
- explicit treatment of action costs without collapsing useful exploration
- turn-level critic signals at the tool-call granularity

Candidate ingredients include:

- patch-aware policy updates
- tool-type-conditioned value estimation
- cost-aware clipping or auxiliary objectives
- trajectory segmentation around edits and test outcomes
- compacted sub-trace training for very long rollouts
- online anti-hack signals that penalize invalid actions without discarding entire trajectories
- explicit belief blocks for long-horizon task state tracking

### 2. Benchmarking

The benchmark layer defines how PAPPO should be evaluated.

The project is especially interested in tasks where an agent must:

- inspect an unfamiliar repository
- localize a bug or missing feature
- make one or more edits
- run tests or other verification tools
- recover from failed intermediate attempts

Evaluation should go beyond final task success and include:

- resolved issue rate
- number of tool calls
- token usage
- wall-clock cost
- test efficiency
- trajectory length
- stability across random seeds

### 3. Interpretability

The interpretability layer studies why the agent behaves the way it does.

Key questions include:

- When does the policy decide that another test run is worth the cost?
- Which hidden states predict a successful patch versus wasted editing?
- How do value estimates change after reading code, seeing a stack trace, or applying a patch?
- Can internal signals explain over-searching, over-testing, or premature stopping?

This part of the project is meant to connect RL for coding agents with mechanistic and behavioral interpretability.

## Value Modeling Hypothesis

PAPPO treats the value model as a first-class research object.

For long-horizon coding agents, the critic should estimate more than final task success. A strong value model should also track patch progress, tool utility, future cost, compaction quality, and failure risk. This enables more stable advantage estimation, better use of compacted sub-traces, more robust handling of abnormal or hacked actions, and clearer interpretation of tool-use behavior.

The working hypothesis is:

- GRPO-style group-relative learning is strong for short, comparable rollouts.
- Long-horizon coding agents produce irregular trajectories with variable length, tool calls, edits, tests, and compaction boundaries.
- In this setting, PAPPO should use critic-based turn-level advantages over individual compacted sub-traces.
- Tool-call turns are a natural credit-assignment unit: they are coarser than tokens, finer than full traces, and often contain enough semantic content for incremental evaluation.
- Optional belief blocks can expose the agent's current task model, making both actor behavior and critic estimates easier to study.
- Invalid or hacked tool calls should be handled online by penalizing the bad action while preserving the rest of the rollout when possible.

## Research Goals

The current goals for PAPPO are:

- define a clear PPO-style objective tailored to tool-using coding agents
- establish a benchmark protocol that exposes long-horizon agent behavior
- study the trade-off between solution quality and tool expenditure
- analyze policy behavior with interpretable intermediate signals
- produce a small, credible, and reproducible research artifact

## Non-Goals

PAPPO is not currently intended to be:

- a generic RL framework
- a broad agent runtime platform
- a production orchestration system
- a repository of many unrelated RL baselines

## Early Roadmap

### Phase 1

- formalize the coding-agent MDP or POMDP
- define the PAPPO objective and value-learning setup
- select the initial benchmark tasks

### Phase 2

- implement a minimal training stack
- compare against strong PPO-style and heuristic baselines
- characterize behavior under different tool-use budgets

### Phase 3

- run interpretability studies on trained agents
- analyze tool-use decisions and patch dynamics
- refine the method based on benchmark failures

## Guiding Principle

PAPPO is based on the belief that coding-agent RL should not only optimize for final correctness. It should also learn how to spend attention, tool use, and iteration budget in a principled way, while remaining understandable enough to study.
