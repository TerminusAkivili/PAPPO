from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_realrepo_pappo_ppo_tiny_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_realrepo_pappo_ppo.py",
            "--model",
            "tiny-local",
            "--backend",
            "scripted",
            "--manifest",
            "data/realrepofix_100_manifest.jsonl",
            "--train-limit",
            "2",
            "--eval-limit",
            "1",
            "--updates",
            "1",
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["status"] == "pappo_ppo_completed"
    assert parsed["updates"] == 1
    assert parsed["seed"] == 7
    assert parsed["train_tasks"] == 2
    assert parsed["eval_tasks"] == 1
    assert parsed["normalize_advantages"] is True
    assert "base_metrics" in parsed
    assert parsed["update_reports"][0]["status"] == "pappo_ppo_update_completed"
    assert "policy_loss" in parsed["update_reports"][0]
    assert "eval_metrics" in parsed["update_reports"][0]
    assert (output_dir / "update_000" / "ppo_samples.jsonl").exists()
    assert (output_dir / "update_000" / "metrics.json").exists()
    assert (output_dir / "update_000" / "adapter" / "adapter_config.json").exists()
    assert (output_dir / "eval" / "update_000_metrics.json").exists()


def test_realrepo_pappo_ppo_supports_grouped_train_rollouts(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo_grouped"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_realrepo_pappo_ppo.py",
            "--model",
            "tiny-local",
            "--backend",
            "scripted",
            "--manifest",
            "data/realrepofix_100_manifest.jsonl",
            "--train-limit",
            "2",
            "--eval-limit",
            "1",
            "--updates",
            "1",
            "--num-rollouts-per-task",
            "2",
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)

    assert parsed["num_rollouts_per_task"] == 2
    assert parsed["local_prior"] == 0.25
    assert parsed["update_reports"][0]["critic_samples"] == 4
    assert parsed["update_reports"][0]["critic_type"] == "group_mean"
    assert parsed["update_reports"][0]["advantage_prior"] == 0.25


def test_realrepo_pappo_ppo_supports_reward_mode_and_action_scope(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo_trace_all_tools"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_realrepo_pappo_ppo.py",
            "--model",
            "tiny-local",
            "--backend",
            "scripted",
            "--manifest",
            "data/realrepofix_100_manifest.jsonl",
            "--train-limit",
            "2",
            "--eval-limit",
            "1",
            "--updates",
            "1",
            "--reward-mode",
            "trace",
            "--action-scope",
            "all_tools",
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)

    assert parsed["reward_mode"] == "trace"
    assert parsed["action_scope"] == "all_tools"
    assert parsed["update_reports"][0]["reward_mode"] == "trace"
    assert parsed["update_reports"][0]["action_scope"] == "all_tools"
    assert parsed["update_reports"][0]["critic_samples"] > 2


def test_realrepo_pappo_ppo_supports_rloo_critic_mode(tmp_path: Path) -> None:
    output_dir = tmp_path / "ppo_rloo"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_realrepo_pappo_ppo.py",
            "--model",
            "tiny-local",
            "--backend",
            "scripted",
            "--manifest",
            "data/realrepofix_100_manifest.jsonl",
            "--train-limit",
            "2",
            "--eval-limit",
            "1",
            "--updates",
            "1",
            "--num-rollouts-per-task",
            "2",
            "--critic-mode",
            "rloo",
            "--output-dir",
            str(output_dir),
            "--seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)

    assert parsed["critic_mode"] == "rloo"
    assert parsed["update_reports"][0]["critic_type"] == "rloo"
    assert parsed["update_reports"][0]["critic_samples"] == 4
