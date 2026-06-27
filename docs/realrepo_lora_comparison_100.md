# RealRepoFix-100 Non-PPO Baselines

This report records the deterministic-v2 non-PPO baseline package used by the
final PAPPO true-PPO comparison.

## Setup

- Model: `Qwen/Qwen2.5-Coder-7B-Instruct`
- Backend: Hugging Face local model
- Seeds: 0, 1, 2
- Split per seed: shuffled 70 train tasks and 30 held-out eval tasks
- Train rollouts: 2 stochastic rollouts per train task
- Eval decoding: stochastic, temperature 0.7
- LoRA epochs: 5
- LoRA learning rate: `5e-5`
- Trainable parameters per adapter: 2,523,136

Artifacts:

- `data/realrepo_lora_comparison_100_seed0_deterministic_v2/report.json`
- `data/realrepo_lora_comparison_100_seed1_deterministic_v2/report.json`
- `data/realrepo_lora_comparison_100_seed2_deterministic_v2/report.json`

## Per-Seed Success

| Seed | Base | Trace | Token Broadcast | PAPPO Turn | PAPPO-v2 | PAPPO-v3 | GRPO-lite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.8333 | 0.8333 | 0.8000 | 0.8000 | 0.8333 | 0.8333 | 0.8333 |
| 1 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 | 0.9000 |
| 2 | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 0.8667 | 0.8667 |
| Mean | 0.8667 | 0.8667 | 0.8556 | 0.8556 | 0.8667 | 0.8667 | 0.8667 |

## Interpretation

The deterministic-v2 weighted-LoRA baselines do not improve over the base model.
The best non-PPO baseline for the final comparison is therefore 0.8667 mean
held-out success. This is the reference score used by
`docs/pappo_true_ppo_results.md`, where PAPPO true PPO reaches 1.0000 mean
held-out success.
