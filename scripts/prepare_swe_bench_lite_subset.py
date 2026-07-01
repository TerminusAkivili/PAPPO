"""Prepare a SWE-bench Lite subset manifest with local repository checkouts."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def _load_swe_bench_lite(split: str, limit: int) -> list[dict]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"missing datasets dependency: {exc}") from exc

    return list(load_dataset("princeton-nlp/SWE-bench_Lite", split=f"{split}[:{limit}]"))


def _patch_source_file(patch: str) -> str:
    for line in patch.splitlines():
        if line.startswith("diff --git "):
            parts = line.split()
            if len(parts) >= 4 and parts[3].startswith("b/"):
                return parts[3][2:]
        if line.startswith("+++ b/"):
            return line[len("+++ b/") :].strip()
    return ""


def _json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _test_command(instance: dict) -> list[str]:
    fail_to_pass = _json_list(instance.get("FAIL_TO_PASS"))
    if fail_to_pass:
        return ["python", "-m", "pytest", "-q", *fail_to_pass]
    return ["python", "-m", "pytest", "-q"]


def _ensure_mirror(repo: str, mirror_dir: Path) -> None:
    mirror_dir = mirror_dir.resolve()
    if mirror_dir.exists():
        return
    mirror_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "clone",
            "--bare",
            "--filter=blob:none",
            f"https://github.com/{repo}.git",
            str(mirror_dir),
        ]
    )


def _ensure_checkout(repo: str, base_commit: str, repo_dir: Path, mirror_dir: Path) -> None:
    repo_dir = repo_dir.resolve()
    mirror_dir = mirror_dir.resolve()
    _ensure_mirror(repo, mirror_dir)
    _run(["git", "-C", str(mirror_dir), "fetch", "--depth=1", "origin", base_commit])
    if repo_dir.exists() and (repo_dir / ".git").exists():
        current = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        if current.returncode == 0 and current.stdout.strip() == base_commit:
            return
        _run(["git", "-C", str(mirror_dir), "worktree", "remove", "--force", str(repo_dir)])
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "git",
            "-C",
            str(mirror_dir),
            "worktree",
            "add",
            "--detach",
            str(repo_dir),
            base_commit,
        ]
    )


def _task_row(instance: dict, index: int, repo_root: Path | None) -> dict:
    instance_id = str(instance.get("instance_id") or f"swe-lite-{index:04d}")
    repo = str(instance.get("repo") or "")
    base_commit = str(instance.get("base_commit") or "")
    repo_dir = ""
    if repo_root is not None:
        mirror_dir = repo_root / "_mirrors" / f"{_safe_name(repo)}.git"
        repo_dir_path = repo_root / _safe_name(instance_id) / "repo"
        _ensure_checkout(repo, base_commit, repo_dir_path, mirror_dir)
        repo_dir = str(repo_dir_path)
    return {
        "task_id": instance_id,
        "benchmark": "swe_bench_lite",
        "repo": repo,
        "base_commit": base_commit,
        "repo_dir": repo_dir,
        "source_file": _patch_source_file(str(instance.get("patch") or "")),
        "problem_statement": str(instance.get("problem_statement") or ""),
        "test_patch": str(instance.get("test_patch") or ""),
        "patch": str(instance.get("patch") or ""),
        "source": "princeton-nlp/SWE-bench_Lite",
        "test_command": _test_command(instance),
        "status": "ready",
    }


def _load_instances(path: Path | None, limit: int) -> list[dict]:
    if path is None:
        return [
            {
                "instance_id": f"swe-lite-placeholder-{index:04d}",
                "repo": "",
                "base_commit": "",
                "problem_statement": "Placeholder row; replace by loading SWE-bench Lite.",
            }
            for index in range(limit)
        ]
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except Exception as exc:
                raise ValueError(f"failed to parse {path}:{line_number}") from exc
            if len(rows) >= limit:
                break
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--split", default="test")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path("data/swe_bench_lite_repos"))
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write rows without cloning/checking out repositories.",
    )
    args = parser.parse_args()

    if args.download:
        instances = _load_swe_bench_lite(args.split, args.limit)
    else:
        instances = _load_instances(args.source_jsonl, args.limit)

    repo_root = None if args.manifest_only else args.repo_root
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for index, instance in enumerate(instances):
            handle.write(json.dumps(_task_row(instance, index, repo_root), sort_keys=True) + "\n")

    payload = {
        "status": "swe_bench_lite_manifest_prepared",
        "output": str(args.output),
        "tasks": len(instances),
        "source_jsonl": str(args.source_jsonl) if args.source_jsonl else None,
        "repo_root": str(repo_root) if repo_root is not None else None,
        "runner_status": "ready" if repo_root is not None else "manifest_only",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
