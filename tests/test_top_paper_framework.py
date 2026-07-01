from __future__ import annotations

import json
import subprocess
from pathlib import Path

from pappo.top_paper import (
    DEFAULT_TOP_PAPER_CONFIG,
    bootstrap_success_ci,
    build_experiment_plan,
    failure_categories,
    failure_case_examples,
    mechanism_metrics,
    paired_win_loss,
    sign_test_p_value,
)
from pappo.trajectory import AgentEvent, AgentTrajectory


def _trajectory(
    trajectory_id: str,
    *,
    final_reward: float,
    patch_applied: bool,
    pytest_reward: float,
    retries: int = 0,
) -> AgentTrajectory:
    events = [
        AgentEvent(
            kind="message",
            content="repair this repository",
        ),
        AgentEvent(
            kind="tool_call",
            tool_name="edit",
            content="replace samplepkg/logic.py",
            cost=2.0,
            metadata={"generated_tokens": 12 + retries},
        ),
        AgentEvent(
            kind="tool_result",
            tool_name="edit",
            content="def fixed():\n    return True\n",
            cost=0.5,
            metadata={"patch_applied": patch_applied},
        ),
        AgentEvent(
            kind="tool_call",
            tool_name="run_test",
            content="python -m pytest -q",
            cost=2.0,
        ),
        AgentEvent(
            kind="tool_result",
            tool_name="run_test",
            content="pytest output",
            cost=1.0,
            metadata={"pytest_reward": pytest_reward},
        ),
    ]
    for index in range(retries):
        events.extend(
            [
                AgentEvent(
                    kind="tool_call",
                    tool_name="edit",
                    content=f"retry patch {index}",
                    cost=2.0,
                    metadata={"generated_tokens": 5},
                ),
                AgentEvent(
                    kind="tool_result",
                    tool_name="edit",
                    content="bad patch",
                    cost=0.5,
                    metadata={"patch_applied": False},
                ),
            ]
        )
    return AgentTrajectory(
        trajectory_id=trajectory_id,
        events=tuple(events),
        final_reward=final_reward,
        metadata={"task_id": trajectory_id},
    )


def test_top_paper_plan_covers_requested_qwen_matrix() -> None:
    plan = build_experiment_plan(DEFAULT_TOP_PAPER_CONFIG)

    assert plan.models == ["Qwen/Qwen2.5-Coder-7B-Instruct"]
    assert [benchmark.name for benchmark in plan.benchmarks] == [
        "realrepofix_100",
        "realrepofix_300",
        "swe_bench_lite_subset",
    ]
    assert plan.seeds == [0, 1, 2, 3, 4]
    assert any(
        "scripts/generate_realrepofix.py" in " ".join(command)
        and "--num-tasks" in command
        and "300" in command
        for command in plan.setup_commands
    )
    assert any(
        "scripts/prepare_swe_bench_lite_subset.py" in " ".join(command)
        and "--output"
        in command
        for command in plan.setup_commands
    )
    assert any(
        gap["benchmark"] == "swe_bench_lite_subset"
        and gap["status"] == "needs_external_swe_data"
        for gap in plan.benchmark_gaps
    )

    method_names = [method.name for method in plan.methods]
    for required in [
        "base_model",
        "reward_weighted_lora",
        "ppo_trace_reward",
        "ppo_token_broadcast_reward",
        "ppo_pappo_turn_local_reward",
        "pappo_without_grouped_baseline",
        "pappo_without_local_prior",
        "pappo_without_kl_ref_control",
        "pappo_sft_weighted_lora",
        "edit_only_vs_all_tool",
        "dpo_ipo_preference",
        "grpo_rloo_grouped_rollout",
        "react_retry_test_heuristic",
        "reflexion_self_repair",
    ]:
        assert required in method_names

    runnable = [job for job in plan.jobs if job.command]
    missing = [job for job in plan.jobs if job.status == "needs_runner"]
    assert runnable
    assert not missing
    assert any("scripts/run_realrepo_pappo_ppo.py" in " ".join(job.command or []) for job in runnable)
    runnable_methods = {job.method for job in runnable}
    assert "ppo_trace_reward" in runnable_methods
    assert "ppo_token_broadcast_reward" in runnable_methods
    assert "standard_ppo_terminal_reward" in runnable_methods
    assert "edit_only_vs_all_tool" in runnable_methods
    assert "react_retry_test_heuristic" in runnable_methods
    assert "reflexion_self_repair" in runnable_methods
    assert "grpo_rloo_grouped_rollout" in runnable_methods
    assert "dpo_ipo_preference" in runnable_methods
    dpo_command = next(job.command for job in runnable if job.method == "dpo_ipo_preference")
    assert dpo_command is not None
    assert "scripts/run_preference_lora_baseline.py" in " ".join(dpo_command)
    swe_base_command = next(
        job.command
        for job in runnable
        if job.benchmark == "swe_bench_lite_subset" and job.method == "base_model"
    )
    assert swe_base_command is not None
    assert "scripts/run_swe_bench_lite_pilot.py" in " ".join(swe_base_command)
    assert "--backend" in swe_base_command
    assert "hf" in swe_base_command
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in swe_base_command
    swe_react_command = next(
        job.command
        for job in runnable
        if job.benchmark == "swe_bench_lite_subset"
        and job.method == "react_retry_test_heuristic"
    )
    assert swe_react_command is not None
    assert "react_retry" in swe_react_command
    swe_reflexion_command = next(
        job.command
        for job in runnable
        if job.benchmark == "swe_bench_lite_subset"
        and job.method == "reflexion_self_repair"
    )
    assert swe_reflexion_command is not None
    assert "reflexion" in swe_reflexion_command
    swe_lora_command = next(
        job.command
        for job in runnable
        if job.benchmark == "swe_bench_lite_subset"
        and job.method == "pappo_sft_weighted_lora"
    )
    assert swe_lora_command is not None
    assert "scripts/run_swe_bench_lora_baseline.py" in " ".join(swe_lora_command)
    assert "pappo_turn_v2" in swe_lora_command
    swe_ppo_command = next(
        job.command
        for job in runnable
        if job.benchmark == "swe_bench_lite_subset"
        and job.method == "ppo_pappo_turn_local_reward"
    )
    assert swe_ppo_command is not None
    assert "scripts/run_swe_bench_pappo_ppo.py" in " ".join(swe_ppo_command)
    assert "pappo_turn_local" in swe_ppo_command


def test_mechanism_metrics_quantify_edit_behavior() -> None:
    trajectories = [
        _trajectory("a", final_reward=1.0, patch_applied=True, pytest_reward=1.0),
        _trajectory("b", final_reward=0.0, patch_applied=True, pytest_reward=0.0, retries=1),
        _trajectory("c", final_reward=0.0, patch_applied=False, pytest_reward=0.0),
    ]

    metrics = mechanism_metrics(trajectories)

    assert metrics["trajectory_count"] == 3
    assert metrics["edit_success_rate"] == 1 / 4
    assert metrics["patch_apply_rate"] == 2 / 4
    assert metrics["failed_patch_rate"] == 1 / 4
    assert metrics["test_pass_after_edit_rate"] == 1 / 3
    assert metrics["avg_retries"] == 1 / 3
    assert metrics["avg_test_calls"] == 1.0
    assert metrics["avg_patch_tokens"] == 10.5
    assert metrics["tool_pattern:edit>run_test"] == 2
    assert metrics["tool_pattern:edit>run_test>edit"] == 1


def test_bootstrap_ci_and_paired_win_loss_are_deterministic() -> None:
    ci = bootstrap_success_ci([1.0, 1.0, 0.0, 1.0], iterations=200, seed=123)
    assert ci["mean"] == 0.75
    assert 0.0 <= ci["low"] <= ci["mean"] <= ci["high"] <= 1.0

    comparison = paired_win_loss(
        baseline={"a": 1.0, "b": 0.0, "c": 0.0},
        candidate={"a": 1.0, "b": 1.0, "c": 0.0},
    )
    assert comparison == {
        "pairs": 3,
        "wins": 1,
        "losses": 0,
        "ties": 2,
        "win_rate": 1 / 3,
        "loss_rate": 0.0,
    }
    assert sign_test_p_value(wins=3, losses=0) == 0.25


def test_failure_categories_and_case_examples() -> None:
    baseline = [
        _trajectory("fixed", final_reward=1.0, patch_applied=True, pytest_reward=1.0),
        _trajectory("improved", final_reward=0.0, patch_applied=True, pytest_reward=0.0),
        _trajectory("tie_fail", final_reward=0.0, patch_applied=False, pytest_reward=0.0),
    ]
    candidate = [
        _trajectory("fixed", final_reward=1.0, patch_applied=True, pytest_reward=1.0),
        _trajectory("improved", final_reward=1.0, patch_applied=True, pytest_reward=1.0),
        _trajectory("tie_fail", final_reward=0.0, patch_applied=False, pytest_reward=0.0),
    ]

    categories = failure_categories(baseline)
    cases = failure_case_examples(baseline=baseline, candidate=candidate, limit=2)

    assert categories == {
        "failed_patch": 1,
        "no_patch": 1,
        "test_failed_after_patch": 1,
    }
    assert cases["improved"][0]["task_id"] == "improved"
    assert cases["improved"][0]["baseline_reward"] == 0.0
    assert cases["improved"][0]["candidate_reward"] == 1.0
    assert cases["persistent_failures"][0]["task_id"] == "tie_fail"


def test_top_paper_plan_script_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "plan.json"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/plan_top_paper_experiments.py",
            "--output",
            str(output),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)

    assert parsed["status"] == "top_paper_plan_created"
    assert parsed["job_count"] > 0
    assert parsed["runnable_job_count"] > 0
    assert parsed["needs_runner_count"] == 0
    assert output.exists()
    saved = json.loads(output.read_text())
    assert saved["job_count"] == parsed["job_count"]


def test_top_paper_audit_script_reports_runnable_gaps(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/audit_top_paper_plan.py",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)

    assert parsed["status"] == "top_paper_audit_completed"
    assert parsed["runnable_job_count"] == 225
    assert parsed["needs_runner_count"] == 0
    assert parsed["by_benchmark"]["realrepofix_100"]["runnable"] == 75
    assert parsed["by_benchmark"]["realrepofix_300"]["runnable"] == 75
    assert parsed["by_benchmark"]["swe_bench_lite_subset"]["runnable"] == 75
    assert parsed["by_benchmark"]["swe_bench_lite_subset"]["needs_runner"] == 0
    assert parsed["gaps"][0]["status"] == "needs_external_swe_data"
    assert parsed["needs_runner_jobs"] == []
    assert output.exists()


def test_top_paper_runner_filters_jobs_and_writes_shell(tmp_path: Path) -> None:
    shell_path = tmp_path / "jobs.sh"
    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/run_top_paper_jobs.py",
            "--benchmark",
            "swe_bench_lite_subset",
            "--method",
            "ppo_pappo_turn_local_reward",
            "--seed",
            "0",
            "--dry-run",
            "--write-shell",
            str(shell_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)

    assert parsed["status"] == "top_paper_jobs_planned"
    assert parsed["dry_run"] is True
    assert parsed["selected_job_count"] == 1
    assert parsed["jobs"][0]["benchmark"] == "swe_bench_lite_subset"
    assert parsed["jobs"][0]["method"] == "ppo_pappo_turn_local_reward"
    assert parsed["jobs"][0]["seed"] == 0
    assert shell_path.exists()
    shell_text = shell_path.read_text(encoding="utf-8")
    assert "scripts/run_swe_bench_pappo_ppo.py" in shell_text


def test_top_paper_analysis_script_reports_ci_and_mechanisms(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.jsonl"
    candidate_path = tmp_path / "candidate.jsonl"
    output = tmp_path / "analysis.json"

    baseline = [
        _trajectory("a", final_reward=1.0, patch_applied=True, pytest_reward=1.0),
        _trajectory("b", final_reward=0.0, patch_applied=True, pytest_reward=0.0),
    ]
    candidate = [
        _trajectory("a", final_reward=1.0, patch_applied=True, pytest_reward=1.0),
        _trajectory("b", final_reward=1.0, patch_applied=True, pytest_reward=1.0),
    ]

    def write_jsonl(path: Path, trajectories: list[AgentTrajectory]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for trajectory in trajectories:
                handle.write(
                    json.dumps(
                        {
                            "trajectory_id": trajectory.trajectory_id,
                            "events": [
                                {
                                    "kind": event.kind,
                                    "content": event.content,
                                    "tool_name": event.tool_name,
                                    "cost": event.cost,
                                    "metadata": dict(event.metadata),
                                }
                                for event in trajectory.events
                            ],
                            "final_reward": trajectory.final_reward,
                            "metadata": dict(trajectory.metadata),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    write_jsonl(baseline_path, baseline)
    write_jsonl(candidate_path, candidate)

    completed = subprocess.run(
        [
            ".venv/bin/python",
            "scripts/analyze_top_paper_results.py",
            "--baseline-rollouts",
            str(baseline_path),
            "--candidate-rollouts",
            str(candidate_path),
            "--bootstrap-iterations",
            "100",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(completed.stdout)

    assert parsed["status"] == "top_paper_analysis_completed"
    assert parsed["candidate_success_ci"]["mean"] == 1.0
    assert parsed["baseline_success_ci"]["mean"] == 0.5
    assert parsed["paired_win_loss"]["wins"] == 1
    assert parsed["paired_sign_test_p"] == 1.0
    assert parsed["failure_categories"]["failed_patch"] == 0
    assert parsed["case_examples"]["improved"][0]["task_id"] == "b"
    assert parsed["mechanism_metrics"]["failed_patch_rate"] == 0.0
    assert output.exists()
