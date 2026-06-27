# PAPPO arXiv Readiness

Current status:

```text
ready_candidate
```

The current public package supports a carefully scoped feasibility preprint
around the RealRepoFix-100 true-PPO result.

## Supported Claim

```text
Tool-call turns are an effective credit-assignment unit for coding-agent PPO.
In a local Qwen2.5-Coder-7B RealRepoFix-100 run, PAPPO-PPO improves held-out
task success over the base model and the published non-PPO LoRA baselines.
```

This is a local single-model result. It should not be stated as a broad claim
that PAPPO improves every coding model, agent runtime, or benchmark family.

## Evidence In This Repository

- RealRepoFix-100 task set and manifest.
- Deterministic three-seed non-PPO baseline reports.
- Three-seed true online PAPPO-PPO reports.
- Focused tests for PPO loss mechanics, rollout sample construction, grouped
  critic behavior, RealRepoFix runner behavior, and deterministic rollout
  seeding.
- Documentation of the final experiment in
  `docs/pappo_true_ppo_results.md`.

## Core Numbers

| Metric | Best non-PPO | PAPPO-PPO |
| --- | ---: | ---: |
| Mean held-out success | 0.8667 | 1.0000 |
| Failed edit rate | 0.1333 | 0.0000 |
| Average tool cost | 8.50 | 8.50 |
| Repeated test rate | 0.0000 | 0.0000 |

PAPPO-PPO wins on all three seeds and reaches 1.0000 success on both update 3
and update 4 for every seed.

## Before Broader Claims

Before claiming broad PAPPO gains for trained full LLM coding agents, add:

- independent reruns or an external coding benchmark
- ablations for the local prior, grouped baseline, KL strength, rollout count,
  and critic design
- true-PPO trajectory case studies
- a clear negative-result boundary for settings where the gain does not hold

## Decision

Ready for a feasibility preprint if the title, abstract, and claims explicitly
scope the result to this local RealRepoFix-style Qwen2.5-Coder-7B true-PPO
study.
