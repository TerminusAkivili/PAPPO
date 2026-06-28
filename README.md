# PAPPO

PAPPO is a research prototype for **Patch-Aware Proximal Policy Optimization**
on tool-using coding agents.

This repository currently publishes the final RealRepoFix-100 true-PPO evidence
package: the runnable PPO loop, the RealRepoFix task set, deterministic baseline
reports, three-seed PAPPO-PPO reports, and focused tests for the PPO mechanics.

## Current Result

Model: `Qwen/Qwen2.5-Coder-7B-Instruct`

Benchmark: RealRepoFix-100 with 70 train tasks and 30 held-out eval tasks per
seed.

| Method | Mean Success | Failed Edit | Avg Tool Cost | Repeated Test |
| --- | ---: | ---: | ---: | ---: |
| Best non-PPO baseline | 0.8667 | 0.1333 | 8.50 | 0.0000 |
| PAPPO true PPO | 1.0000 | 0.0000 | 8.50 | 0.0000 |

PAPPO-PPO wins on all three seeds and keeps 1.0000 held-out success for both
update 3 and update 4. See
[`docs/pappo_true_ppo_results.md`](docs/pappo_true_ppo_results.md).

## Citation And Archival Trace

- Core experiment commit:
  `5d345d06c735dd263ab0a85f24200c52dd9993e5`
- Release tag: `v0.1.0-trueppo`
- Citation metadata: [`CITATION.cff`](CITATION.cff)
- Zenodo metadata: [`.zenodo.json`](.zenodo.json)
- Release notes:
  [`RELEASE_NOTES_v0.1.0-trueppo.md`](RELEASE_NOTES_v0.1.0-trueppo.md)
- Software Heritage snapshot:
  `swh:1:snp:51a2836ced67734772ba3c5f0bfa69d4b48e1427`

The Zenodo DOI should be added here after the GitHub-Zenodo integration archives
the release.

## License

PAPPO is released under the Apache License 2.0. See [`LICENSE`](LICENSE).

## Key Paths

- PPO objective and logprob handling: `pappo/ppo_training.py`
- PPO sample extraction: `pappo/ppo_rollout.py`
- Turn/group critic: `pappo/turn_critic.py`
- RealRepoFix task and agent backends: `pappo/realrepofix.py`,
  `pappo/llm_agent_pilot.py`
- True-PPO experiment runner: `scripts/run_realrepo_pappo_ppo.py`
- Baseline runner used for comparison: `scripts/run_realrepo_lora_comparison.py`
- RealRepoFix manifest: `data/realrepofix_100_manifest.jsonl`
- Technical report draft: `docs/pappo_technical_report.md`
- Final result reports:
  - `data/realrepo_lora_comparison_100_seed0_deterministic_v2/report.json`
  - `data/realrepo_lora_comparison_100_seed1_deterministic_v2/report.json`
  - `data/realrepo_lora_comparison_100_seed2_deterministic_v2/report.json`
  - `data/realrepo_pappo_ppo_100_seed0_grouped_prior_lr1e4_nonorm_updates5_v2/report.json`
  - `data/realrepo_pappo_ppo_100_seed1_grouped_prior_lr1e4_nonorm_updates5/report.json`
  - `data/realrepo_pappo_ppo_100_seed2_grouped_prior_lr1e4_nonorm_updates5/report.json`

Large adapter weights, exploratory rollout dumps, and intermediate artifacts are
intentionally not tracked.

## Verify The Published Result

Run the focused test suite:

```bash
python -m pytest -q
```

Audit the committed result reports:

```bash
python - <<'PY'
import json
from pathlib import Path

baseline_paths = {
    0: Path('data/realrepo_lora_comparison_100_seed0_deterministic_v2/report.json'),
    1: Path('data/realrepo_lora_comparison_100_seed1_deterministic_v2/report.json'),
    2: Path('data/realrepo_lora_comparison_100_seed2_deterministic_v2/report.json'),
}
ppo_paths = {
    0: Path('data/realrepo_pappo_ppo_100_seed0_grouped_prior_lr1e4_nonorm_updates5_v2/report.json'),
    1: Path('data/realrepo_pappo_ppo_100_seed1_grouped_prior_lr1e4_nonorm_updates5/report.json'),
    2: Path('data/realrepo_pappo_ppo_100_seed2_grouped_prior_lr1e4_nonorm_updates5/report.json'),
}

rows = []
for seed in [0, 1, 2]:
    baseline = json.loads(baseline_paths[seed].read_text())
    ppo = json.loads(ppo_paths[seed].read_text())
    metrics = {'base': baseline['base_metrics'], **baseline['metrics']}
    best_name, best = max(metrics.items(), key=lambda item: item[1]['success_rate'])
    final = ppo['update_reports'][-1]['eval_metrics']
    rows.append((seed, best_name, best, final, ppo))

mean_best = sum(row[2]['success_rate'] for row in rows) / len(rows)
mean_ppo = sum(row[3]['success_rate'] for row in rows) / len(rows)
wins = sum(row[3]['success_rate'] > row[2]['success_rate'] for row in rows)

print('mean_best_non_ppo_success', round(mean_best, 4))
print('mean_pappo_ppo_success', round(mean_ppo, 4))
print('delta', round(mean_ppo - mean_best, 4))
print('wins', wins, '/', len(rows))

assert mean_ppo - mean_best >= 0.05
assert wins >= 2
assert all(row[4]['train_tasks'] == 70 and row[4]['eval_tasks'] == 30 for row in rows)
assert all(row[4]['num_rollouts_per_task'] == 2 for row in rows)
assert all(row[4]['update_reports'][-1]['eval_metrics']['failed_edit_rate'] <= row[2]['failed_edit_rate'] for row in rows)
assert all(row[4]['update_reports'][-1]['eval_metrics']['avg_tool_cost'] <= row[2]['avg_tool_cost'] for row in rows)
assert all(row[4]['update_reports'][-1]['eval_metrics']['repeated_test_rate'] <= row[2]['repeated_test_rate'] for row in rows)
assert all(row[4]['update_reports'][-1]['eval_metrics']['success_rate'] == 1.0 for row in rows)
assert all(row[4]['update_reports'][-2]['eval_metrics']['success_rate'] == 1.0 for row in rows)
print('AUDIT_PASS')
PY
```

Expected summary:

```text
mean_best_non_ppo_success 0.8667
mean_pappo_ppo_success 1.0
delta 0.1333
wins 3 / 3
AUDIT_PASS
```

## Reproduce The Final Experiment

The full run requires a local/downloadable Qwen2.5-Coder-7B-Instruct model and
a GPU with enough memory for LoRA training.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_XET=1 \
python scripts/run_realrepo_pappo_ppo.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --backend hf \
  --manifest data/realrepofix_100_manifest.jsonl \
  --train-limit 70 \
  --eval-limit 30 \
  --updates 5 \
  --num-rollouts-per-task 2 \
  --max-new-tokens 384 \
  --temperature 0.7 \
  --max-length 768 \
  --learning-rate 1e-4 \
  --ppo-epochs 1 \
  --local-prior 0.25 \
  --no-normalize-advantages \
  --seed 0 \
  --output-dir data/repro_realrepo_pappo_ppo_seed0 \
  --local-files-only
```

Run the same command with `--seed 1` and `--seed 2` for the three-seed result.

## Scope

This is a local single-model research finding. It supports the claim that
turn-local PAPPO credit assignment becomes useful when embedded in a true PPO
loop. It is not a claim that PAPPO improves every coding model, benchmark, or
agent runtime.
