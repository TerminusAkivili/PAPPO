# Notes

## SWE-bench Lite data preparation

The experiment framework now has runnable commands for the SWE-bench Lite subset, but the SWE manifest must still be populated with real local repository checkouts before running those jobs.

Each SWE-bench Lite manifest row should include:

- `repo_dir`: path to the local checked-out repository.
- `source_file`: repository-relative source file path used for prompt context.
- `patch`: candidate or gold unified diff patch for patch-based training/evaluation paths.
- `test_command`: command used to validate the task in that checkout.

This is an external data preparation requirement, not a missing framework or runner component.

## First high-value run

Before running the full 225-job matrix, prioritize the smallest set that directly tests the main PAPPO claim:

- Benchmark: `realrepofix_100`
- Seeds: `0, 1, 2, 3, 4`
- Methods:
  - `ppo_trace_reward`
  - `ppo_token_broadcast_reward`
  - `ppo_pappo_turn_local_reward`

This is 15 jobs total: `RealRepoFix-100 x 5 seeds x 3 PPO reward modes`.

Dry-run the queue first:

```bash
.venv/bin/python scripts/run_top_paper_jobs.py \
  --benchmark realrepofix_100 \
  --method ppo_trace_reward \
  --method ppo_token_broadcast_reward \
  --method ppo_pappo_turn_local_reward \
  --dry-run \
  --write-shell data/top_paper_qwen/run_realrepofix100_core15.sh
```

Then execute the generated shell script when ready:

```bash
bash data/top_paper_qwen/run_realrepofix100_core15.sh
```

Decision rule:

- If these 15 jobs show that `ppo_pappo_turn_local_reward` consistently beats or matches the two reward baselines (`ppo_trace_reward` and `ppo_token_broadcast_reward`) across seeds, the main PAPPO claim is worth scaling.
- The RealRepoFix-100 core run saturated: all three PPO reward modes reached perfect final eval metrics. This validates the PPO/LoRA training path, but it does not yet separate PAPPO from trace/token reward baselines.
- Do not run the remaining 210 jobs immediately. First run the same core comparison on the harder external benchmark slice below.

## Second high-value run

Run only the SWE-bench Lite subset core comparison:

- Benchmark: `swe_bench_lite_subset`
- Seeds: `0, 1, 2, 3, 4`
- Methods:
  - `ppo_trace_reward`
  - `ppo_token_broadcast_reward`
  - `ppo_pappo_turn_local_reward`

This is 15 jobs total: `SWE-bench Lite subset x 5 seeds x 3 PPO reward modes`.

Dry-run and export the queue:

```bash
.venv/bin/python scripts/run_top_paper_jobs.py \
  --benchmark swe_bench_lite_subset \
  --method ppo_trace_reward \
  --method ppo_token_broadcast_reward \
  --method ppo_pappo_turn_local_reward \
  --dry-run \
  --write-shell data/top_paper_qwen/run_swe_lite_core15.sh
```

Then execute:

```bash
bash data/top_paper_qwen/run_swe_lite_core15.sh
```

Decision rule:

- If PAPPO separates from trace/token reward on SWE-bench Lite, scale to the broader top-paper matrix.
- If SWE-bench Lite is blocked by missing local repo checkout fields in the manifest, finish the SWE manifest preparation first; this is a data-prep blocker, not a training-framework blocker.

Current status:

- The SWE-bench Lite runner and queue generator are part of the repository.
- The concrete SWE-bench Lite manifest is treated as local data preparation,
  because it contains machine-specific repository checkout paths.
- Before launching the SWE core comparison, prepare
  `data/swe_bench_lite_subset_manifest.jsonl` with real local checkouts and
  verify that every row has `repo_dir`, `source_file`, `patch`, and
  `test_command`.
- Do not scale to the broader matrix until the SWE core comparison completes
  cleanly and separates the three PPO reward modes.
