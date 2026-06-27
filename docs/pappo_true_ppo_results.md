# PAPPO True PPO RealRepoFix-100 Results

This report records the first RealRepoFix-100 true-PPO PAPPO run that clears
the current core-method gate.

## Setup

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Backend: Hugging Face local model
- Split per seed: 70 train tasks and 30 held-out eval tasks
- Seeds: 0, 1, 2
- PPO updates: 5
- Train rollouts: 2 stochastic rollouts per train task per update
- Eval decoding: stochastic, temperature 0.7
- LoRA learning rate: `1e-4`
- PPO epochs per update: 1
- Local PAPPO prior: 0.25
- Advantage normalization: disabled
- Max new tokens: 384
- Max training length: 768

The PPO loop stores old-policy and reference logprobs, trains with an
action-masked clipped PPO objective plus KL control, uses grouped turn-level
baselines, updates adapters cumulatively, re-rolls out after each update, and
evaluates every checkpoint on held-out tasks.

## Artifacts

Baseline weighted-LoRA comparison artifacts:

- `data/realrepo_lora_comparison_100_seed0_deterministic_v2/report.json`
- `data/realrepo_lora_comparison_100_seed1_deterministic_v2/report.json`
- `data/realrepo_lora_comparison_100_seed2_deterministic_v2/report.json`

PAPPO-PPO artifacts:

- `data/realrepo_pappo_ppo_100_seed0_grouped_prior_lr1e4_nonorm_updates5_v2/report.json`
- `data/realrepo_pappo_ppo_100_seed1_grouped_prior_lr1e4_nonorm_updates5/report.json`
- `data/realrepo_pappo_ppo_100_seed2_grouped_prior_lr1e4_nonorm_updates5/report.json`

## Three-Seed Result

The best non-PPO baseline includes the base model and all weighted-LoRA
conditions from the deterministic v2 comparison. Including or excluding the
older PAPPO-weighted LoRA variants does not change the best baseline score.

| Seed | Best Non-PPO Success | PAPPO-PPO Base | PAPPO-PPO Final | Delta | Failed Edit: Best -> PPO | Cost: Best -> PPO | Repeated Test: Best -> PPO |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8333 | 0.8333 | 1.0000 | +0.1667 | 0.1667 -> 0.0000 | 8.50 -> 8.50 | 0.0000 -> 0.0000 |
| 1 | 0.9000 | 0.9000 | 1.0000 | +0.1000 | 0.1000 -> 0.0000 | 8.50 -> 8.50 | 0.0000 -> 0.0000 |
| 2 | 0.8667 | 0.8667 | 1.0000 | +0.1333 | 0.1333 -> 0.0000 | 8.50 -> 8.50 | 0.0000 -> 0.0000 |
| Mean | 0.8667 | 0.8667 | 1.0000 | +0.1333 | 0.1333 -> 0.0000 | 8.50 -> 8.50 | 0.0000 -> 0.0000 |

Checkpoint success rates:

| Seed | Update 0 | Update 1 | Update 2 | Update 3 | Update 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8333 | 0.8333 | 0.9000 | 1.0000 | 1.0000 |
| 1 | 0.9000 | 0.9000 | 0.9000 | 1.0000 | 1.0000 |
| 2 | 0.8667 | 0.9000 | 0.9000 | 1.0000 | 1.0000 |

Checkpoint KL values:

| Seed | Update 0 | Update 1 | Update 2 | Update 3 | Update 4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.000000 | 0.000048 | 0.000223 | 0.003136 | 0.035791 |
| 1 | 0.000000 | 0.000058 | 0.000189 | 0.002219 | 0.020127 |
| 2 | 0.000000 | 0.000062 | 0.000240 | 0.002476 | 0.019943 |

## Gate Check

| Requirement | Status |
| --- | --- |
| RealRepoFix-100 scale with 70/30 train/eval split | Pass |
| 3 seeds | Pass |
| True PPO mechanics: old logprobs, clipped objective, KL/ref, baseline, rollout-after-update, checkpoint eval | Pass |
| Mean success at least 5pp above best non-PPO baseline | Pass: +13.33pp |
| Wins on at least 2 of 3 seeds | Pass: 3/3 |
| Failed edit does not regress | Pass: 13.33% -> 0.00% |
| Repeated test does not regress | Pass: 0.00% -> 0.00% |
| Average tool cost does not regress | Pass: 8.50 -> 8.50 |
| Advantage is not a single cherry-picked spike | Pass: updates 3 and 4 both hold 1.0000 on all seeds |

## Interpretation

The weighted-LoRA RealRepoFix-100 result showed that reward-weighted SFT was too
blunt: it often matched or underperformed the base model. The true PPO loop
changes that result. With grouped turn-local PAPPO advantages, KL/reference
control, clipped updates, and fresh rollouts after every update, PAPPO-PPO
achieves a stable held-out success improvement across all three seeds without
increasing tool cost or repeated tests.

This supports the central third-stage claim: PAPPO's turn-local credit
assignment becomes useful when it is embedded in a real PPO loop, not merely
used as a supervised LoRA weight.

The result is still a local single-model research finding. Before making a
broader paper claim, the next checks should include independent reruns, a
larger or external coding benchmark, and ablations for the local prior,
grouped baseline, KL weight, rollout count, and critic design.
