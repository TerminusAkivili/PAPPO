"""SWE-bench Lite subset utilities for PAPPO trajectory collection."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from pappo.trajectory import AgentEvent, AgentTrajectory


@dataclass(frozen=True)
class SWEBenchLiteTask:
    """One SWE-bench Lite-style repository repair task."""

    task_id: str
    repo_dir: Path
    problem_statement: str
    patch: str
    test_command: tuple[str, ...]
    source_file: Path | None = None
    base_commit: str = ""
    repo: str = ""


def _row_test_command(row: dict) -> tuple[str, ...]:
    command = row.get("test_command")
    if isinstance(command, list):
        return tuple(str(item) for item in command)
    if isinstance(command, str) and command:
        return tuple(command.split())
    return ("python", "-m", "pytest", "-q")


def load_swe_bench_lite_tasks(manifest: Path, limit: int | None = None) -> list[SWEBenchLiteTask]:
    """Load SWE-bench Lite subset tasks from a JSONL manifest."""

    tasks: list[SWEBenchLiteTask] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and len(tasks) >= limit:
                break
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except Exception as exc:
                raise ValueError(f"failed to parse {manifest}:{line_number}") from exc
            repo_dir = row.get("repo_dir") or row.get("local_repo_dir")
            if not repo_dir:
                raise ValueError(f"missing repo_dir in {manifest}:{line_number}")
            source_file = row.get("source_file")
            tasks.append(
                SWEBenchLiteTask(
                    task_id=str(row.get("task_id") or row.get("instance_id")),
                    repo_dir=Path(repo_dir),
                    problem_statement=str(
                        row.get("problem_statement")
                        or row.get("issue")
                        or "Repair the repository."
                    ),
                    patch=str(row.get("patch") or ""),
                    test_command=_row_test_command(row),
                    source_file=Path(source_file) if source_file else None,
                    base_commit=str(row.get("base_commit") or ""),
                    repo=str(row.get("repo") or ""),
                )
            )
    return tasks


def _copy_repo(task: SWEBenchLiteTask, work_root: Path) -> Path:
    work_dir = work_root / task.task_id
    if work_dir.exists():
        shutil.rmtree(work_dir)
    ignore = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc")
    shutil.copytree(task.repo_dir, work_dir, ignore=ignore)
    return work_dir


def _run_command(command: tuple[str, ...], cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _apply_patch(work_dir: Path, patch: str, timeout_seconds: int) -> tuple[bool, str]:
    if not patch.strip():
        return False, "empty patch"
    completed = subprocess.run(
        ["patch", "-p1"],
        input=patch,
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )
    return completed.returncode == 0, completed.stdout + completed.stderr


def extract_unified_diff(text: str) -> str | None:
    """Extract a unified diff from raw model output."""

    stripped = text.strip()
    if "```" in stripped:
        pieces = stripped.split("```")
        for piece in pieces:
            candidate = piece.strip()
            if candidate.startswith("diff"):
                candidate = candidate.split("\n", 1)[1] if "\n" in candidate else ""
            if candidate.startswith("--- ") and "\n+++ " in candidate:
                return candidate.strip() + "\n"
    if stripped.startswith("--- ") and "\n+++ " in stripped:
        return stripped + ("\n" if not stripped.endswith("\n") else "")
    return None


def evaluate_patch_task(
    task: SWEBenchLiteTask,
    *,
    work_root: Path,
    timeout_seconds: int = 30,
) -> AgentTrajectory:
    """Apply a candidate patch, run tests, and emit a PAPPO trajectory."""

    work_root.mkdir(parents=True, exist_ok=True)
    work_dir = _copy_repo(task, work_root)
    events: list[AgentEvent] = [
        AgentEvent(
            kind="message",
            content=task.problem_statement,
            metadata={
                "benchmark": "swe_bench_lite",
                "task_id": task.task_id,
            },
        )
    ]

    patch_applied, patch_output = _apply_patch(work_dir, task.patch, timeout_seconds)
    events.append(
        AgentEvent(
            kind="tool_call",
            tool_name="edit",
            content=task.patch,
            cost=1.0,
            metadata={"generated_tokens": len(task.patch.split())},
        )
    )
    events.append(
        AgentEvent(
            kind="tool_result",
            tool_name="edit",
            content=patch_output,
            cost=0.0,
            metadata={"patch_applied": patch_applied},
        )
    )

    pytest_reward = 0.0
    test_output = ""
    return_code = -1
    if patch_applied:
        completed = _run_command(task.test_command, work_dir, timeout_seconds)
        return_code = completed.returncode
        test_output = completed.stdout + completed.stderr
        pytest_reward = 1.0 if completed.returncode == 0 else 0.0
    events.append(
        AgentEvent(
            kind="tool_call",
            tool_name="run_test",
            content=" ".join(task.test_command),
            cost=1.0,
        )
    )
    events.append(
        AgentEvent(
            kind="tool_result",
            tool_name="run_test",
            content=test_output,
            cost=0.0,
            metadata={
                "pytest_reward": pytest_reward,
                "return_code": return_code,
            },
        )
    )

    return AgentTrajectory(
        trajectory_id=task.task_id,
        events=tuple(events),
        final_reward=pytest_reward,
        metadata={
            "benchmark": "swe_bench_lite",
            "task_id": task.task_id,
            "repo": task.repo,
            "repo_dir": str(work_dir),
            "base_commit": task.base_commit,
        },
    )


@dataclass
class SWEBenchLiteLLMBackend:
    """Generate SWE-bench Lite candidate patches with a HF causal LM."""

    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    max_new_tokens: int = 768
    temperature: float = 0.0
    adapter_path: Path | None = None
    local_files_only: bool = False
    name: str = "swe_qwen2.5_coder_7b"
    is_full_llm: bool = True

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(f"missing HF runtime dependency: {exc}") from exc

        self._torch = torch
        dtype = torch.bfloat16 if torch.cuda.is_available() else "auto"
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=self.local_files_only,
        )
        if self.adapter_path is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                is_trainable=False,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _source_context(self, task: SWEBenchLiteTask) -> str:
        if task.source_file is None:
            return "<source file unavailable>"
        path = task.repo_dir / task.source_file
        if not path.exists():
            return f"{task.source_file}: <missing>"
        return f"{task.source_file}\n```\n{path.read_text(encoding='utf-8')[:6000]}\n```"

    def _prompt(self, task: SWEBenchLiteTask) -> str:
        return (
            "You are a coding agent repairing a Python repository.\n"
            "Return only a unified diff patch. Do not include prose.\n\n"
            f"Repository: {task.repo}\n"
            f"Task: {task.task_id}\n"
            f"Issue:\n{task.problem_statement}\n\n"
            f"Relevant source:\n{self._source_context(task)}\n\n"
            f"Test command: {' '.join(task.test_command)}\n"
        )

    def _retry_prompt(
        self,
        task: SWEBenchLiteTask,
        previous_output: str,
        previous_test_output: str,
    ) -> str:
        return self._prompt(task) + (
            "\nThe previous attempt failed. Try one more unified diff patch.\n"
            "Previous model output:\n"
            f"```\n{previous_output[-2000:]}\n```\n"
            "Previous test output:\n"
            f"```\n{previous_test_output[-4000:]}\n```\n"
        )

    def _generate(self, prompt: str) -> tuple[str, dict[str, int | str]]:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = prompt
        inputs = self.tokenizer(text, return_tensors="pt")
        device = getattr(self.model, "device", None)
        if device is not None:
            inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = int(inputs["input_ids"].shape[-1])
        do_sample = self.temperature > 0.0
        with self._torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=do_sample,
                temperature=self.temperature if do_sample else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text, {
            "prompt_tokens": input_length,
            "generated_tokens": int(generated_ids.shape[-1]),
            "model_path": self.model_path,
            "raw_prompt_text": text,
        }

    def repair(
        self,
        task: SWEBenchLiteTask,
        *,
        work_root: Path,
        timeout_seconds: int = 30,
    ) -> AgentTrajectory:
        prompt = self._prompt(task)
        generated_text, generation_metadata = self._generate(prompt)
        patch = extract_unified_diff(generated_text)
        candidate_task = SWEBenchLiteTask(
            task_id=task.task_id,
            repo_dir=task.repo_dir,
            problem_statement=task.problem_statement,
            patch=patch or "",
            test_command=task.test_command,
            source_file=task.source_file,
            base_commit=task.base_commit,
            repo=task.repo,
        )
        trajectory = evaluate_patch_task(
            candidate_task,
            work_root=work_root,
            timeout_seconds=timeout_seconds,
        )
        events = list(trajectory.events)
        for index, event in enumerate(events):
            if event.kind == "tool_call" and event.tool_name == "edit":
                events[index] = AgentEvent(
                    kind=event.kind,
                    tool_name=event.tool_name,
                    content=event.content,
                    cost=event.cost,
                    metadata=dict(generation_metadata),
                )
            if event.kind == "tool_result" and event.tool_name == "edit":
                events[index] = AgentEvent(
                    kind=event.kind,
                    tool_name=event.tool_name,
                    content=event.content,
                    cost=event.cost,
                    metadata={
                        **dict(event.metadata),
                        "backend": self.name,
                        "model_path": self.model_path,
                        "raw_generated_text": generated_text,
                    },
                )
        return AgentTrajectory(
            trajectory_id=f"{task.task_id}-{self.name}",
            events=tuple(events),
            final_reward=trajectory.final_reward,
            metadata={
                **dict(trajectory.metadata),
                "agent_backend": self.name,
                "full_llm": self.is_full_llm,
                "model_path": self.model_path,
                "patch_applied": any(
                    event.tool_name == "edit"
                    and bool(event.metadata.get("patch_applied"))
                    for event in events
                ),
            },
        )


class SWEBenchLiteReActBackend(SWEBenchLiteLLMBackend):
    """SWE-bench Lite backend that retries once after failed patch/test feedback."""

    name: str = "swe_react_retry"
    max_attempts: int = 2

    def repair(
        self,
        task: SWEBenchLiteTask,
        *,
        work_root: Path,
        timeout_seconds: int = 30,
    ) -> AgentTrajectory:
        events: list[AgentEvent] = []
        final_reward = 0.0
        patch_applied = False
        previous_output = ""
        previous_test_output = ""

        for attempt_index in range(self.max_attempts):
            prompt = (
                self._prompt(task)
                if attempt_index == 0
                else self._retry_prompt(task, previous_output, previous_test_output)
            )
            generated_text, generation_metadata = self._generate(prompt)
            patch = extract_unified_diff(generated_text)
            candidate_task = SWEBenchLiteTask(
                task_id=f"{task.task_id}-attempt{attempt_index + 1}",
                repo_dir=task.repo_dir,
                problem_statement=task.problem_statement,
                patch=patch or "",
                test_command=task.test_command,
                source_file=task.source_file,
                base_commit=task.base_commit,
                repo=task.repo,
            )
            attempt = evaluate_patch_task(
                candidate_task,
                work_root=work_root,
                timeout_seconds=timeout_seconds,
            )
            final_reward = attempt.final_reward
            previous_output = generated_text
            previous_test_output = "\n".join(
                event.content
                for event in attempt.events
                if event.tool_name == "run_test" and event.kind == "tool_result"
            )
            for event in attempt.events:
                if event.kind == "message" and events:
                    continue
                metadata = dict(event.metadata)
                metadata["attempt"] = attempt_index + 1
                if event.tool_name == "edit" and event.kind == "tool_call":
                    metadata.update(generation_metadata)
                if event.tool_name == "edit" and event.kind == "tool_result":
                    metadata.update(
                        {
                            "backend": self.name,
                            "model_path": self.model_path,
                            "raw_generated_text": generated_text,
                        }
                    )
                    patch_applied = patch_applied or bool(metadata.get("patch_applied"))
                events.append(
                    AgentEvent(
                        kind=event.kind,
                        tool_name=event.tool_name,
                        content=event.content,
                        cost=event.cost,
                        metadata=metadata,
                    )
                )
            if final_reward > 0.0:
                break

        return AgentTrajectory(
            trajectory_id=f"{task.task_id}-{self.name}",
            events=tuple(events),
            final_reward=final_reward,
            metadata={
                "benchmark": "swe_bench_lite",
                "task_id": task.task_id,
                "repo": task.repo,
                "agent_backend": self.name,
                "full_llm": self.is_full_llm,
                "model_path": self.model_path,
                "patch_applied": patch_applied,
            },
        )


class SWEBenchLiteReflexionBackend(SWEBenchLiteReActBackend):
    """SWE-bench Lite retry backend with an explicit reflection prompt."""

    name: str = "swe_reflexion"

    def _retry_prompt(
        self,
        task: SWEBenchLiteTask,
        previous_output: str,
        previous_test_output: str,
    ) -> str:
        return self._prompt(task) + (
            "\nPrevious attempt failed. Reflect on the failure and return only "
            "a corrected unified diff patch.\n"
            "Previous model output:\n"
            f"```\n{previous_output[-2000:]}\n```\n"
            "Failure feedback:\n"
            f"```\n{previous_test_output[-4000:]}\n```\n"
        )


def write_trajectories(path: Path, trajectories: list[AgentTrajectory]) -> None:
    """Write SWE-bench Lite PAPPO trajectories as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(asdict(trajectory), sort_keys=True) + "\n")
