from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pappo.preference_pairs import build_preference_pairs, load_preference_pairs
from scripts.run_realrepo_lora_comparison import load_jsonl_trajectories


def test_build_preference_pairs_from_mixed_realrepofix_trajectories() -> None:
    trajectories = load_jsonl_trajectories(Path("data/realrepofix_100_mixed_trajectories.jsonl"))

    pairs = build_preference_pairs(trajectories, limit=3)

    assert len(pairs) == 3
    for pair in pairs:
        assert pair.chosen_reward > pair.rejected_reward
        assert pair.prompt
        assert pair.chosen
        assert pair.rejected
        assert pair.group_key


def test_preference_pair_script_writes_jsonl_and_summary(tmp_path: Path) -> None:
    output = tmp_path / "pairs.jsonl"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/build_preference_pairs.py",
            "--trajectories",
            "data/realrepofix_100_mixed_trajectories.jsonl",
            "--output",
            str(output),
            "--limit",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    pairs = load_preference_pairs(output)

    assert parsed["status"] == "preference_pairs_completed"
    assert parsed["pairs"] == 4
    assert output.exists()
    assert len(pairs) == 4
    assert pairs[0].chosen_reward > pairs[0].rejected_reward


def test_preference_lora_baseline_trains_tiny_adapter(tmp_path: Path) -> None:
    pairs_path = tmp_path / "pairs.jsonl"
    output_dir = tmp_path / "preference_lora"
    pairs_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "pair_id": "pair-1",
                        "group_key": "task-1",
                        "prompt": "<task> fix it",
                        "chosen": "<tool_call name=\"edit\"> def fixed(): return True </tool_call>",
                        "rejected": "<tool_call name=\"edit\"> def broken(): return False </tool_call>",
                        "chosen_reward": 1.0,
                        "rejected_reward": 0.0,
                        "reward_delta": 1.0,
                        "chosen_trajectory_id": "task-1-success",
                        "rejected_trajectory_id": "task-1-failure",
                    }
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_preference_lora_baseline.py",
            "--model",
            "tiny-local",
            "--pairs",
            str(pairs_path),
            "--output-dir",
            str(output_dir),
            "--max-length",
            "64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["status"] == "preference_lora_completed"
    assert parsed["method"] == "dpo_ipo_like"
    assert parsed["pairs"] == 1
    assert parsed["examples"] == 2
    assert (output_dir / "adapter" / "adapter_config.json").exists()
