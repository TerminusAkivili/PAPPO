"""Executable RealRepoFix benchmark generation and reward running."""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pappo.trajectory import AgentEvent, AgentTrajectory, MESSAGE, TOOL_CALL, TOOL_RESULT


@dataclass(frozen=True)
class RealRepoFixTask:
    """One executable repository repair task."""

    task_id: str
    repo_dir: Path
    issue: str
    source_file: Path
    test_file: Path
    buggy_source: str
    fixed_source: str
    test_command: tuple[str, ...]
    template: str


@dataclass(frozen=True)
class PytestReward:
    """Reward result from executing pytest."""

    reward: float
    returncode: int
    stdout: str
    stderr: str


def _write_task_files(task_dir: Path, source: str, test: str) -> None:
    package_dir = task_dir / "repo" / "samplepkg"
    tests_dir = task_dir / "repo" / "tests"
    package_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("", encoding="utf-8")
    (package_dir / "logic.py").write_text(source, encoding="utf-8")
    (tests_dir / "test_logic.py").write_text(test, encoding="utf-8")


def _template_parser(index: int) -> tuple[str, str, str, str]:
    buggy = """def parse_csv_line(line):\n    return line.split(',')\n"""
    fixed = """import csv\n\n\ndef parse_csv_line(line):\n    return next(csv.reader([line]))\n"""
    test = """from samplepkg.logic import parse_csv_line\n\n\ndef test_parse_csv_line_handles_quoted_comma():\n    assert parse_csv_line('a,\"b,c\",d') == ['a', 'b,c', 'd']\n"""
    return (
        f"CSV parser task {index}: handle quoted commas.",
        buggy,
        fixed,
        test,
    )


def _template_cache(index: int) -> tuple[str, str, str, str]:
    buggy = """def cache_key(namespace, key):\n    return key\n"""
    fixed = """def cache_key(namespace, key):\n    return f'{namespace}:{key}'\n"""
    test = """from samplepkg.logic import cache_key\n\n\ndef test_cache_key_includes_namespace():\n    assert cache_key('users', '42') != cache_key('orders', '42')\n    assert cache_key('users', '42') == 'users:42'\n"""
    return (
        f"Cache task {index}: namespace must affect cache keys.",
        buggy,
        fixed,
        test,
    )


def _template_serializer(index: int) -> tuple[str, str, str, str]:
    buggy = """def serialize_user(data):\n    return {key: value for key, value in data.items() if value}\n"""
    fixed = """def serialize_user(data):\n    return {key: value for key, value in data.items() if value is not None}\n"""
    test = """from samplepkg.logic import serialize_user\n\n\ndef test_serialize_user_keeps_false_values():\n    assert serialize_user({'name': 'Ada', 'active': False, 'age': 0}) == {'name': 'Ada', 'active': False, 'age': 0}\n"""
    return (
        f"Serializer task {index}: preserve explicit false-like values.",
        buggy,
        fixed,
        test,
    )


def _template_retry(index: int) -> tuple[str, str, str, str]:
    buggy = """def should_retry(error_name):\n    return True\n"""
    fixed = """def should_retry(error_name):\n    return error_name not in {'ValidationError', 'PermissionError'}\n"""
    test = """from samplepkg.logic import should_retry\n\n\ndef test_should_retry_skips_permanent_errors():\n    assert should_retry('TimeoutError') is True\n    assert should_retry('ValidationError') is False\n"""
    return (
        f"Retry task {index}: permanent errors should not retry.",
        buggy,
        fixed,
        test,
    )


def _template_auth(index: int) -> tuple[str, str, str, str]:
    buggy = """def is_expired(now, exp):\n    return now > exp\n"""
    fixed = """def is_expired(now, exp):\n    return now >= exp\n"""
    test = """from samplepkg.logic import is_expired\n\n\ndef test_is_expired_at_boundary():\n    assert is_expired(10, 10) is True\n    assert is_expired(9, 10) is False\n"""
    return (
        f"Auth task {index}: expiry boundary should be closed.",
        buggy,
        fixed,
        test,
    )


TEMPLATES = (
    ("csv_parser", _template_parser),
    ("cache_namespace", _template_cache),
    ("serializer_false_values", _template_serializer),
    ("retry_permanent_errors", _template_retry),
    ("auth_expiry_boundary", _template_auth),
)


def generate_realrepofix_tasks(
    root: Path,
    num_tasks: int = 100,
    seed: int = 0,
) -> list[RealRepoFixTask]:
    """Generate executable Python repair tasks under `root`."""

    if num_tasks < 1:
        raise ValueError("num_tasks must be positive")
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    tasks: list[RealRepoFixTask] = []

    for index in range(num_tasks):
        template_name, template = TEMPLATES[index % len(TEMPLATES)]
        issue, buggy, fixed, test = template(index)
        task_id = f"realrepofix-{index:04d}-{template_name}"
        task_dir = root / task_id
        if task_dir.exists():
            # Keep generation deterministic without relying on stale files.
            for path in sorted(task_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        _write_task_files(task_dir, buggy, test)
        task = RealRepoFixTask(
            task_id=task_id,
            repo_dir=task_dir / "repo",
            issue=issue,
            source_file=Path("samplepkg/logic.py"),
            test_file=Path("tests/test_logic.py"),
            buggy_source=buggy,
            fixed_source=fixed,
            test_command=("python", "-m", "pytest", "-q"),
            template=template_name,
        )
        tasks.append(task)

    rng.shuffle(tasks)
    return tasks


def run_pytest_reward(task: RealRepoFixTask, timeout_seconds: int = 10) -> PytestReward:
    """Run pytest for a RealRepoFix task and return reward."""

    completed = subprocess.run(
        task.test_command,
        cwd=task.repo_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return PytestReward(
        reward=1.0 if completed.returncode == 0 else 0.0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def apply_expected_fix(task: RealRepoFixTask) -> None:
    """Apply the expected source fix for a task."""

    (task.repo_dir / task.source_file).write_text(task.fixed_source, encoding="utf-8")


def export_realrepofix_trajectories(
    tasks: list[RealRepoFixTask],
) -> list[AgentTrajectory]:
    """Export successful scripted trajectories backed by real pytest rewards."""

    trajectories: list[AgentTrajectory] = []
    for task in tasks:
        source_before = (task.repo_dir / task.source_file).read_text(encoding="utf-8")
        before = run_pytest_reward(task)
        apply_expected_fix(task)
        after = run_pytest_reward(task)
        events = (
            AgentEvent(kind=MESSAGE, content=task.issue),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="read_file",
                content=str(task.source_file),
                cost=1.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="read_file",
                content=source_before,
                cost=0.5,
                metadata={"pytest_before_reward": before.reward},
            ),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="edit",
                content=f"replace {task.source_file} with expected fix",
                cost=2.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="edit",
                content=task.fixed_source,
                cost=0.5,
                metadata={"patch_applied": True},
            ),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="run_test",
                content=" ".join(task.test_command),
                cost=2.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="run_test",
                content=f"pytest returncode={after.returncode}\n{after.stdout}",
                cost=1.0,
                metadata={"pytest_reward": after.reward},
            ),
        )
        trajectories.append(
            AgentTrajectory(
                trajectory_id=task.task_id,
                events=events,
                final_reward=after.reward,
                metadata={
                    "template": task.template,
                    "benchmark": "realrepofix",
                    "repo_dir": str(task.repo_dir),
                },
            )
        )

    return trajectories


def _reset_buggy_source(task: RealRepoFixTask) -> None:
    (task.repo_dir / task.source_file).write_text(task.buggy_source, encoding="utf-8")


def export_realrepofix_mixed_trajectories(
    tasks: list[RealRepoFixTask],
) -> list[AgentTrajectory]:
    """Export success and failure trajectories backed by real pytest rewards."""

    trajectories: list[AgentTrajectory] = []
    for task in tasks:
        _reset_buggy_source(task)
        source_before = (task.repo_dir / task.source_file).read_text(encoding="utf-8")
        failure = run_pytest_reward(task)
        failure_events = (
            AgentEvent(kind=MESSAGE, content=task.issue),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="read_file",
                content=str(task.source_file),
                cost=1.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="read_file",
                content=source_before,
                cost=0.5,
                metadata={"pytest_before_reward": failure.reward},
            ),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="edit",
                content=f"inspect {task.source_file} but leave bug unchanged",
                cost=1.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="edit",
                content="no effective patch applied",
                cost=0.5,
                metadata={"patch_applied": False},
            ),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="run_test",
                content=" ".join(task.test_command),
                cost=2.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="run_test",
                content=f"pytest returncode={failure.returncode}\n{failure.stdout}",
                cost=1.0,
                metadata={"pytest_reward": failure.reward},
            ),
        )
        trajectories.append(
            AgentTrajectory(
                trajectory_id=f"{task.task_id}-failure",
                events=failure_events,
                final_reward=failure.reward,
                metadata={
                    "template": task.template,
                    "benchmark": "realrepofix",
                    "mode": "failure",
                    "repo_dir": str(task.repo_dir),
                },
            )
        )

        _reset_buggy_source(task)
        trajectories.extend(export_realrepofix_trajectories([task]))
        _reset_buggy_source(task)

    return trajectories
