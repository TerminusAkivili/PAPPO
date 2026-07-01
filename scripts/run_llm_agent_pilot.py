"""Run a RealRepoFix LLM coding-agent pilot when a local model is available."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pappo.llm_agent_pilot import (  # noqa: E402
    HuggingFaceRepairBackend,
    ReflexionRepairBackend,
    ReActRetryRepairBackend,
    ScriptedRepairBackend,
    discover_local_models,
)
from pappo.realrepofix import RealRepoFixTask, TEMPLATES  # noqa: E402


def _manifest_exists(path: Path) -> bool:
    return path.exists() and path.is_file()


def _model_available(model_path: str) -> bool:
    path = Path(model_path)
    if path.exists():
        return True
    # Treat bare HF ids as unavailable here unless explicitly downloaded. This
    # script should not silently start network downloads in a benchmark run.
    return False


def _template_sources_for_manifest_row(row: dict) -> tuple[str, str, str]:
    template_name = row["template"]
    task_index = int(row["task_id"].split("-")[1])
    for name, template in TEMPLATES:
        if name == template_name:
            _issue, buggy, fixed, _test = template(task_index)
            return buggy, fixed, _test
    raise ValueError(f"unknown template: {template_name}")


def _load_tasks(manifest: Path, limit: int) -> list[RealRepoFixTask]:
    tasks: list[RealRepoFixTask] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(tasks) >= limit:
                break
            row = json.loads(line)
            repo_dir = Path(row["repo_dir"])
            source_file = Path(row["source_file"])
            buggy_source, fixed_source, _test_source = _template_sources_for_manifest_row(row)
            tasks.append(
                RealRepoFixTask(
                    task_id=row["task_id"],
                    repo_dir=repo_dir,
                    issue=row["issue"],
                    source_file=source_file,
                    test_file=Path(row["test_file"]),
                    buggy_source=buggy_source,
                    fixed_source=fixed_source,
                    test_command=tuple(row["test_command"]),
                    template=row["template"],
                )
            )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
    )
    parser.add_argument(
        "--backend",
        choices=["model", "hf", "scripted", "react_retry", "reflexion"],
        default="model",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trajectories", type=Path)
    parser.add_argument("--list-local-models", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only load already cached/local HF model files.",
    )
    args = parser.parse_args()

    if args.list_local_models:
        print(
            json.dumps(
                {
                    "status": "local_models",
                    "models": [
                        {
                            "model_id": model.model_id,
                            "cache_path": model.cache_path,
                            "likely_coder": model.likely_coder,
                        }
                        for model in discover_local_models()
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if not _manifest_exists(args.manifest):
        print(
            json.dumps(
                {
                    "status": "manifest_unavailable",
                    "llm_agent_completed": False,
                    "manifest": str(args.manifest),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.backend == "scripted":
        backend = ScriptedRepairBackend()
        tasks = _load_tasks(args.manifest, args.limit)
        trajectories = [backend.repair(task) for task in tasks]
        if args.trajectories is not None:
            args.trajectories.parent.mkdir(parents=True, exist_ok=True)
            with args.trajectories.open("w", encoding="utf-8") as handle:
                for trajectory in trajectories:
                    handle.write(json.dumps(asdict(trajectory), sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "status": "scripted_agent_completed",
                    "llm_agent_completed": False,
                    "tool_loop_completed": True,
                    "num_trajectories": len(trajectories),
                    "success_rate": (
                        sum(item.final_reward for item in trajectories)
                        / len(trajectories)
                        if trajectories
                        else 0.0
                    ),
                    "trajectories": (
                        str(args.trajectories)
                        if args.trajectories is not None
                        else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    if args.backend == "model" and not _model_available(args.model_path):
        print(
            json.dumps(
                {
                    "status": "model_unavailable",
                    "llm_agent_completed": False,
                    "model_path": args.model_path,
                    "manifest": str(args.manifest),
                    "limit": args.limit,
                    "message": (
                        "Provide a downloaded local coder model path to run "
                        "the full LLM coding-agent pilot."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    backend_cls = HuggingFaceRepairBackend
    if args.backend == "react_retry":
        backend_cls = ReActRetryRepairBackend
    elif args.backend == "reflexion":
        backend_cls = ReflexionRepairBackend

    try:
        backend = backend_cls(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            local_files_only=args.no_download,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "model_load_failed",
                    "llm_agent_completed": False,
                    "tool_loop_completed": False,
                    "model_path": args.model_path,
                    "manifest": str(args.manifest),
                    "limit": args.limit,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    tasks = _load_tasks(args.manifest, args.limit)
    trajectories = []
    run_error = None
    for task in tasks:
        try:
            trajectories.append(backend.repair(task))
        except Exception as exc:
            run_error = {
                "task_id": task.task_id,
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
            break

    if args.trajectories is not None and trajectories:
        args.trajectories.parent.mkdir(parents=True, exist_ok=True)
        with args.trajectories.open("w", encoding="utf-8") as handle:
            for trajectory in trajectories:
                handle.write(json.dumps(asdict(trajectory), sort_keys=True) + "\n")

    completed = bool(trajectories) and run_error is None
    print(
        json.dumps(
            {
                "status": (
                    "llm_agent_completed"
                    if completed
                    else "llm_agent_run_failed"
                ),
                "llm_agent_completed": completed,
                "tool_loop_completed": bool(trajectories),
                "num_trajectories": len(trajectories),
                "success_rate": (
                    sum(item.final_reward for item in trajectories)
                    / len(trajectories)
                    if trajectories
                    else 0.0
                ),
                "model_path": args.model_path,
                "backend": backend.name,
                "trajectories": (
                    str(args.trajectories)
                    if args.trajectories is not None
                    else None
                ),
                "run_error": run_error,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return


if __name__ == "__main__":
    main()
