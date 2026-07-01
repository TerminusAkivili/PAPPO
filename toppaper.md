# PAPPO Top-Tier Paper Upgrade Checklist

This note records what PAPPO still needs before it can be positioned as a
top-tier conference paper rather than a small-scale technical report.

## 1. Multi-Model Validation

The current result uses a single model:
`Qwen/Qwen2.5-Coder-7B-Instruct`.

A top-tier reviewer will ask:

> Is PAPPO effective, or is this result specific to this model and split?

Minimum additions:

- `Qwen2.5-Coder-7B`
- `Qwen2.5-Coder-14B` or `Qwen2.5-Coder-32B`
- At least one external model family, such as DeepSeek-Coder, StarCoder2,
  CodeLlama, or Gemma-Coder
- Ideally one API-agent inference baseline, such as GPT-4.1-mini or Claude
  Haiku-class models, even if it is only used for inference comparison

The goal is not to reach 100% on every model. The goal is to show that the
positive trend is not confined to one model family or one lucky split.

## 2. Multiple Benchmarks And Task Sets

RealRepoFix-100 is a useful benchmark, but one benchmark is not enough for a
top-tier claim.

At least one external benchmark should be added:

- SWE-bench Lite or a SWE-bench Verified subset
- HumanEvalFix, RepoExec, or another repository/code-repair task family
- An expanded RealRepoFix-300 or RealRepoFix-500
- Tasks spanning multiple repositories, languages, and test frameworks

The desired claim is:

> PAPPO stably improves edit behavior across repository repair tasks.

The weaker claim to avoid is:

> PAPPO happens to work on RealRepoFix-100.

## 3. Required Ablations

This is the most important missing piece. PAPPO has several components, and a
reviewer will ask which component actually matters.

| Ablation | Purpose |
| --- | --- |
| Trace reward PPO | Show that PPO alone is not sufficient. |
| Token/broadcast reward PPO | Show that turn-local credit is useful. |
| PAPPO without grouped baseline | Show the contribution of the grouped critic. |
| PAPPO without local prior | Show that the local prior is not arbitrary tuning. |
| PAPPO without KL/ref control | Show that stability comes from true PPO mechanics. |
| PAPPO SFT / weighted LoRA | Show that the signal must be inside a PPO loop. |
| Edit-only vs all-tool optimization | Justify the optimized action scope. |

The most important comparison is:

```text
PPO + trace reward
PPO + token broadcast reward
PPO + PAPPO turn-local reward
```

If PAPPO wins this comparison, the method becomes much more paper-like.

## 4. Statistical Rigor

Three seeds are a good start, but a top-tier paper should provide stronger
statistics.

Recommended additions:

- 5 or 10 seeds
- Confidence intervals
- Bootstrap confidence intervals
- Paired significance tests
- Per-task paired win/loss analysis
- Failure category analysis

Example paper-style statement:

```text
PAPPO improves success by +X pp, with 95% bootstrap CI [a, b],
and paired task-level win/loss = m/n.
```

This turns the result from an experiment log into paper-level evidence.

## 5. Stronger Baselines

The current best non-PPO baseline is not enough for a top-tier submission.

Additional baselines should include:

- Base model
- Reward-weighted SFT / LoRA
- DPO or IPO-like preference baseline, if pairs can be constructed
- GRPO or RLOO grouped-rollout baseline
- Standard PPO with terminal reward
- ReAct-style retry/test heuristic baseline
- Reflexion or self-repair baseline, if feasible

A top-tier reviewer will not only ask:

> Is PAPPO better than non-PPO LoRA?

They will ask:

> Is PAPPO better than reasonable agent-RL baselines?

## 6. Mechanism Analysis

PAPPO's strongest current signal is that edit behavior becomes more reliable.
This should be analyzed directly.

Break down:

- Edit success rate
- Patch apply rate
- Failed patch rate
- Test-pass-after-edit rate
- Number of retries
- Number of test calls
- Token length and patch size
- Tool trajectory pattern changes
- Before/after failure examples

The key point to demonstrate:

> PAPPO does not make the agent more verbose or more expensive. It makes the
> critical edit turn more reliable.

The current result already has a strong mechanism clue: failed edits drop from
12/90 to 0/90. That should be expanded into a full behavioral analysis.

## 7. Tighter Theory And Formalization

PAPPO does not need to become a heavy theory paper, but the method should be
formalized clearly.

Formalize:

- Trajectory
- Tool turn
- Local target
- Baseline
- Masked PPO objective
- Relation to standard PPO
- Why terminal reward broadcasting has higher variance or weaker credit
  assignment

The goal is for readers to see PAPPO as a method, not an empirical trick.

## 8. More Complete Reproducibility Package

The current release package is already useful. For a top-tier artifact, add:

- Exact environment file
- Training command matrix
- Result aggregation script
- One-command audit
- Generated PDF report
- Model download notes
- Expected runtime and cost
- Artifact checksums
- Optional Dockerfile

Hardware differences should be documented carefully, especially across B200,
H200, RTX 4090, and RTX 5090 environments.

## 9. Related Work

A complete related-work section is mandatory for a top-tier submission.

Cover:

- PPO / RLHF / RLAIF
- Process reward models
- Agent trajectory credit assignment
- Tool-use agents
- Coding agents and SWE-bench
- ReAct, Reflexion, SWE-agent
- GRPO / RLOO
- Reward-weighted regression / filtered behavior cloning
- Patch validation and test-based feedback

Without related work, PAPPO can be an arXiv technical report, but it will not
look like a mature conference paper.

## 10. Sharper Claim

The current claim is:

> PAPPO turn-local credit assignment becomes useful inside true PPO.

A sharper but still defensible top-tier claim would be:

> Repository repair agents suffer from action-level credit ambiguity under
> terminal rewards. PAPPO introduces patch-aware turn-local advantages for PPO,
> improving edit reliability and held-out repair success under fixed tool
> budgets.

This positions PAPPO as a credit-assignment method for tool-using coding agents,
not merely as a tuned training recipe.
