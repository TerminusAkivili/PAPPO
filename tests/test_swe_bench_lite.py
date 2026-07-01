from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pappo.swe_bench_lite import (
    SWEBenchLiteLLMBackend,
    SWEBenchLiteReActBackend,
    SWEBenchLiteReflexionBackend,
    evaluate_patch_task,
    load_swe_bench_lite_tasks,
    write_trajectories,
)


def _write_toy_repo(root: Path) -> tuple[Path, Path]:
    repo = root / "toy_repo"
    package = repo / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = package / "logic.py"
    source.write_text(
        "def add_one(value):\n"
        "    return value + 2\n",
        encoding="utf-8",
    )
    tests = repo / "test_logic.py"
    tests.write_text(
        "from pkg.logic import add_one\n\n"
        "def test_add_one():\n"
        "    assert add_one(1) == 2\n",
        encoding="utf-8",
    )
    return repo, source


def _toy_manifest(root: Path, task_id: str, *, patch: str = "") -> Path:
    repo, source = _write_toy_repo(root)
    manifest = root / f"{task_id}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "benchmark": "swe_bench_lite",
                "repo_dir": str(repo),
                "source_file": str(source.relative_to(repo)),
                "problem_statement": "Fix add_one.",
                "patch": patch,
                "test_command": ["python", "-m", "pytest", "-q"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def _fix_patch() -> str:
    return (
        "--- a/pkg/logic.py\n"
        "+++ b/pkg/logic.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add_one(value):\n"
        "-    return value + 2\n"
        "+    return value + 1\n"
    )


def test_swe_bench_lite_runner_applies_patch_and_emits_trajectory(tmp_path: Path) -> None:
    manifest = _toy_manifest(tmp_path, "toy-swe-1", patch=_fix_patch())
    task = load_swe_bench_lite_tasks(manifest, limit=1)[0]
    trajectory = evaluate_patch_task(task, work_root=tmp_path / "work")

    assert trajectory.final_reward == 1.0
    assert trajectory.metadata["benchmark"] == "swe_bench_lite"
    assert trajectory.metadata["task_id"] == "toy-swe-1"
    assert any(
        event.tool_name == "edit"
        and event.metadata.get("patch_applied") is True
        for event in trajectory.events
    )
    assert any(
        event.tool_name == "run_test"
        and event.metadata.get("pytest_reward") == 1.0
        for event in trajectory.events
    )

    output = tmp_path / "trajectories.jsonl"
    write_trajectories(output, [trajectory])
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["final_reward"] == 1.0


def test_swe_bench_lite_react_backend_retries_after_failed_patch(tmp_path: Path) -> None:
    task = load_swe_bench_lite_tasks(_toy_manifest(tmp_path, "toy-swe-react"), limit=1)[0]
    generated = iter(["not a patch", f"```diff\n{_fix_patch()}\n```"])

    backend = SWEBenchLiteReActBackend.__new__(SWEBenchLiteReActBackend)
    backend.model_path = "fake-model"
    backend.max_new_tokens = 128
    backend.temperature = 0.0
    backend.name = "swe_react_retry"
    backend.is_full_llm = True
    backend.max_attempts = 2
    backend._generate = lambda prompt: (  # type: ignore[attr-defined]
        next(generated),
        {"generated_tokens": 12, "raw_prompt_text": prompt},
    )

    trajectory = backend.repair(task, work_root=tmp_path / "react_work")

    assert trajectory.final_reward == 1.0
    assert trajectory.metadata["agent_backend"] == "swe_react_retry"
    assert sum(1 for event in trajectory.events if event.tool_name == "edit") == 4
    assert sum(1 for event in trajectory.events if event.tool_name == "run_test") == 4


def test_swe_bench_lite_reflexion_backend_includes_failure_feedback(tmp_path: Path) -> None:
    task = load_swe_bench_lite_tasks(_toy_manifest(tmp_path, "toy-swe-reflexion"), limit=1)[0]
    prompts: list[str] = []
    generated = iter(["not a patch", f"```diff\n{_fix_patch()}\n```"])

    backend = SWEBenchLiteReflexionBackend.__new__(SWEBenchLiteReflexionBackend)
    backend.model_path = "fake-model"
    backend.max_new_tokens = 128
    backend.temperature = 0.0
    backend.name = "swe_reflexion"
    backend.is_full_llm = True
    backend.max_attempts = 2

    def fake_generate(prompt: str):
        prompts.append(prompt)
        return next(generated), {"generated_tokens": 12, "raw_prompt_text": prompt}

    backend._generate = fake_generate  # type: ignore[attr-defined]

    trajectory = backend.repair(task, work_root=tmp_path / "reflexion_work")

    assert trajectory.final_reward == 1.0
    assert trajectory.metadata["agent_backend"] == "swe_reflexion"
    assert len(prompts) == 2
    assert "Previous attempt failed" in prompts[1]


def test_swe_bench_lite_runner_script_writes_jsonl(tmp_path: Path) -> None:
    repo, source = _write_toy_repo(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "rollouts.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "toy-swe-2",
                "benchmark": "swe_bench_lite",
                "repo_dir": str(repo),
                "source_file": str(source.relative_to(repo)),
                "problem_statement": "Fix add_one.",
                "patch": (
                    "--- a/pkg/logic.py\n"
                    "+++ b/pkg/logic.py\n"
                    "@@ -1,2 +1,2 @@\n"
                    " def add_one(value):\n"
                    "-    return value + 2\n"
                    "+    return value + 1\n"
                ),
                "test_command": ["python", "-m", "pytest", "-q"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_swe_bench_lite_pilot.py",
            "--manifest",
            str(manifest),
            "--limit",
            "1",
            "--work-root",
            str(tmp_path / "work"),
            "--trajectories",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)

    assert parsed["status"] == "swe_bench_lite_pilot_completed"
    assert parsed["num_trajectories"] == 1
    assert parsed["success_rate"] == 1.0
    assert output.exists()


def test_swe_bench_lite_llm_backend_generates_patch_and_runs_tests(tmp_path: Path) -> None:
    repo, source = _write_toy_repo(tmp_path)
    empty_manifest = tmp_path / "manifest.jsonl"
    empty_manifest.write_text("", encoding="utf-8")
    task = load_swe_bench_lite_tasks(empty_manifest, limit=0)
    assert task == []
    generated_patch = (
        "--- a/pkg/logic.py\n"
        "+++ b/pkg/logic.py\n"
        "@@ -1,2 +1,2 @@\n"
        " def add_one(value):\n"
        "-    return value + 2\n"
        "+    return value + 1\n"
    )
    manifest = tmp_path / "llm_manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "task_id": "toy-swe-llm",
                "benchmark": "swe_bench_lite",
                "repo_dir": str(repo),
                "source_file": str(source.relative_to(repo)),
                "problem_statement": "Fix add_one.",
                "patch": "",
                "test_command": ["python", "-m", "pytest", "-q"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    swe_task = load_swe_bench_lite_tasks(manifest, limit=1)[0]

    backend = SWEBenchLiteLLMBackend.__new__(SWEBenchLiteLLMBackend)
    backend.model_path = "fake-model"
    backend.max_new_tokens = 128
    backend.temperature = 0.0
    backend.name = "fake_swe_llm"
    backend.is_full_llm = True
    backend._generate = lambda prompt: (  # type: ignore[attr-defined]
        f"```diff\n{generated_patch}\n```",
        {"generated_tokens": 12, "raw_prompt_text": prompt},
    )
    trajectory = backend.repair(swe_task, work_root=tmp_path / "llm_work")

    assert trajectory.final_reward == 1.0
    assert trajectory.metadata["agent_backend"] == "fake_swe_llm"
    assert any(
        event.tool_name == "edit"
        and "raw_generated_text" in event.metadata
        for event in trajectory.events
    )


def test_swe_bench_lora_baseline_trains_adapter_from_patch_trajectories(tmp_path: Path) -> None:
    manifest = _toy_manifest(tmp_path, "toy-swe-lora", patch=_fix_patch())
    output_dir = tmp_path / "swe_lora"

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_swe_bench_lora_baseline.py",
            "--model",
            "tiny-local",
            "--manifest",
            str(manifest),
            "--train-limit",
            "1",
            "--eval-limit",
            "0",
            "--method",
            "pappo_turn_v2",
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

    assert parsed["status"] == "swe_bench_lora_completed"
    assert parsed["method"] == "pappo_turn_v2"
    assert parsed["train_rollouts_count"] == 1
    assert parsed["train_report"]["examples"] > 0
    assert (output_dir / "pappo_turn_v2" / "adapter" / "adapter_config.json").exists()


def test_swe_bench_pappo_ppo_trains_tiny_adapter(tmp_path: Path) -> None:
    manifest = _toy_manifest(tmp_path, "toy-swe-ppo", patch=_fix_patch())
    output_dir = tmp_path / "swe_ppo"

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_swe_bench_pappo_ppo.py",
            "--model",
            "tiny-local",
            "--manifest",
            str(manifest),
            "--train-limit",
            "1",
            "--eval-limit",
            "0",
            "--updates",
            "1",
            "--reward-mode",
            "pappo_turn_local",
            "--critic-mode",
            "tool_mean",
            "--output-dir",
            str(output_dir),
            "--max-length",
            "64",
            "--no-normalize-advantages",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)

    assert parsed["status"] == "swe_bench_pappo_ppo_completed"
    assert parsed["reward_mode"] == "pappo_turn_local"
    assert parsed["update_reports"][0]["samples"] > 0
    assert (output_dir / "update_000" / "adapter" / "adapter_config.json").exists()


def test_swe_bench_preference_lora_trains_tiny_adapter(tmp_path: Path) -> None:
    manifest = _toy_manifest(tmp_path, "toy-swe-pref", patch=_fix_patch())
    output_dir = tmp_path / "swe_pref"

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_swe_bench_preference_lora.py",
            "--model",
            "tiny-local",
            "--manifest",
            str(manifest),
            "--train-limit",
            "1",
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

    assert parsed["status"] == "swe_bench_preference_lora_completed"
    assert parsed["pairs"] == 1
    assert parsed["examples"] == 2
    assert (output_dir / "adapter" / "adapter_config.json").exists()
