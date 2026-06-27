# PAPPO v0.1.0: True PPO RealRepoFix-100 Reproducibility Package

This release archives the PAPPO true-PPO RealRepoFix-100 reproducibility
package.

## Fixed Reference

- Repository: `https://github.com/TerminusAkivili/PAPPO`
- Tag: `v0.1.0-trueppo`
- Commit: `5d345d06c735dd263ab0a85f24200c52dd9993e5`

## Included

- True PPO implementation with old/ref logprobs, clipped action-masked PPO,
  KL/reference control, grouped turn-local baseline, cumulative LoRA updates,
  rollout-after-update, and checkpoint evaluation.
- RealRepoFix-100 manifest and task files.
- Three deterministic non-PPO baseline reports.
- Three PAPPO-PPO result reports.
- Focused tests for PPO loss mechanics, PPO rollout sample construction,
  grouped critic behavior, RealRepoFix runner behavior, and deterministic
  rollout seeding.

## Main Result

| Metric | Best non-PPO | PAPPO-PPO |
| --- | ---: | ---: |
| Mean held-out success | 0.8667 | 1.0000 |
| Failed edit rate | 0.1333 | 0.0000 |
| Average tool cost | 8.50 | 8.50 |
| Repeated test rate | 0.0000 | 0.0000 |

PAPPO-PPO wins on all three seeds and keeps 1.0000 held-out success for both
update 3 and update 4.

## Scope

This is a local single-model feasibility result for
`Qwen/Qwen2.5-Coder-7B-Instruct` on RealRepoFix-100. It is not a broad claim
that PAPPO improves every coding model, agent runtime, or benchmark family.
