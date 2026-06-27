from __future__ import annotations

import json
import random
import subprocess
from pathlib import Path

import torch

from pappo.trajectory import AgentTrajectory


def test_realrepo_lora_comparison_runs_tiny_scripted_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "comparison"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_realrepo_lora_comparison.py",
            "--model",
            "tiny-local",
            "--backend",
            "scripted",
            "--manifest",
            "data/realrepofix_100_manifest.jsonl",
            "--train-limit",
            "3",
            "--eval-limit",
            "2",
            "--methods",
            "trace",
            "token_broadcast",
            "pappo_turn",
            "pappo_turn_v2",
            "grpo_lite",
            "--eval-base",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["status"] == "realrepo_lora_comparison_completed"
    assert parsed["train_tasks"] == 3
    assert parsed["eval_tasks"] == 2
    assert set(parsed["methods"]) == {
        "trace",
        "token_broadcast",
        "pappo_turn",
        "pappo_turn_v2",
        "grpo_lite",
    }
    assert "base_metrics" in parsed
    for method in parsed["methods"]:
        metrics = parsed["metrics"][method]
        delta = parsed["delta_vs_base"][method]
        assert "success_rate" in metrics
        assert "success_rate" in delta
        assert "avg_tool_cost" in metrics
        assert "edit_success_rate" in metrics
        assert "failed_edit_rate" in metrics
        assert "repeated_test_rate" in metrics
        assert "search_usefulness" in metrics
        assert "retry_task_success_rate" in metrics
        assert (output_dir / method / "adapter" / "adapter_config.json").exists()
    assert (output_dir / "report.md").exists()


def test_rollout_seed_controls_backend_sampling(monkeypatch, tmp_path: Path) -> None:
    from scripts import run_realrepo_lora_comparison as comparison
    import numpy as np

    class RandomBackend:
        def repair(self, task) -> AgentTrajectory:
            return AgentTrajectory(
                trajectory_id=str(task),
                events=(),
                final_reward=1.0,
                metadata={
                    "python_random": random.random(),
                    "numpy_random": float(np.random.random()),
                    "torch_random": float(torch.rand(())),
                },
            )

    monkeypatch.setattr(comparison, "_task_slice", lambda *args, **kwargs: ["a", "b"])
    monkeypatch.setattr(comparison, "_make_backend", lambda *args, **kwargs: RandomBackend())

    first = comparison._rollout(
        backend_name="hf",
        model="dummy",
        adapter_path=None,
        manifest=tmp_path / "manifest.jsonl",
        start=0,
        limit=2,
        max_new_tokens=1,
        temperature=0.7,
        local_files_only=True,
        seed=11,
    )
    second = comparison._rollout(
        backend_name="hf",
        model="dummy",
        adapter_path=None,
        manifest=tmp_path / "manifest.jsonl",
        start=0,
        limit=2,
        max_new_tokens=1,
        temperature=0.7,
        local_files_only=True,
        seed=11,
    )

    assert [trajectory.metadata for trajectory in first] == [
        trajectory.metadata for trajectory in second
    ]


def test_realrepo_lora_comparison_supports_staged_resume(tmp_path: Path) -> None:
    output_dir = tmp_path / "staged"
    common = [
        ".venv/bin/python",
        "scripts/run_realrepo_lora_comparison.py",
        "--model",
        "tiny-local",
        "--backend",
        "scripted",
        "--manifest",
        "data/realrepofix_100_manifest.jsonl",
        "--train-limit",
        "3",
        "--eval-limit",
        "2",
        "--output-dir",
        str(output_dir),
    ]

    rollout = subprocess.run(
        [*common, "--stage", "rollout-train"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(rollout.stdout)["status"] == "rollout_train_completed"

    train = subprocess.run(
        [*common, "--stage", "train-adapter", "--method", "pappo_turn", "--epochs", "2"],
        check=True,
        capture_output=True,
        text=True,
    )
    train_report = json.loads(train.stdout)
    assert train_report["method"] == "pappo_turn"
    assert train_report["epochs"] == 2

    eval_run = subprocess.run(
        [*common, "--stage", "eval-adapter", "--method", "pappo_turn"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "success_rate" in json.loads(eval_run.stdout)

    summary = subprocess.run(
        [
            *common,
            "--stage",
            "summarize",
            "--methods",
            "pappo_turn",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(summary.stdout)
    assert parsed["status"] == "realrepo_lora_comparison_completed"
    assert parsed["methods"] == ["pappo_turn"]
    assert (output_dir / "report.json").exists()


def test_realrepo_lora_comparison_can_group_multiple_train_rollouts_per_task(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "grouped"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_realrepo_lora_comparison.py",
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
            "--num-rollouts-per-task",
            "2",
            "--stage",
            "rollout-train",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["train_tasks"] == 2
    assert parsed["train_rollouts_count"] == 4


def test_realrepo_lora_comparison_supports_explicit_train_eval_split(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "split"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_realrepo_lora_comparison.py",
            "--model",
            "tiny-local",
            "--backend",
            "scripted",
            "--manifest",
            "data/realrepofix_100_manifest.jsonl",
            "--train-start",
            "10",
            "--train-limit",
            "2",
            "--eval-start",
            "50",
            "--eval-limit",
            "1",
            "--seed",
            "7",
            "--stage",
            "rollout-train",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(completed.stdout)
    assert parsed["seed"] == 7
    assert parsed["train_start"] == 10
    assert parsed["eval_start"] == 50
    assert parsed["train_tasks"] == 2


def test_realrepo_lora_comparison_saves_adapter_checkpoints(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "checkpoints"
    common = [
        ".venv/bin/python",
        "scripts/run_realrepo_lora_comparison.py",
        "--model",
        "tiny-local",
        "--backend",
        "scripted",
        "--manifest",
        "data/realrepofix_100_manifest.jsonl",
        "--train-limit",
        "4",
        "--eval-limit",
        "1",
        "--output-dir",
        str(output_dir),
    ]

    subprocess.run([*common, "--stage", "rollout-train"], check=True)
    train = subprocess.run(
        [
            *common,
            "--stage",
            "train-adapter",
            "--method",
            "pappo_turn_v2",
            "--checkpoint-every",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    parsed = json.loads(train.stdout)
    assert parsed["checkpoints"] == 2
    assert (output_dir / "pappo_turn_v2" / "checkpoints" / "after_2" / "adapter_config.json").exists()
    assert (output_dir / "pappo_turn_v2" / "checkpoints" / "after_4" / "adapter_config.json").exists()
