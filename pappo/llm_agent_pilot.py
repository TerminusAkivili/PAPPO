"""Utilities for the full LLM coding-agent pilot."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pappo.realrepofix import (
    RealRepoFixTask,
    apply_expected_fix,
    run_pytest_reward,
)
from pappo.trajectory import AgentEvent, AgentTrajectory, MESSAGE, TOOL_CALL, TOOL_RESULT


PYTHON_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class LocalModelCandidate:
    """One locally cached Hugging Face model candidate."""

    model_id: str
    cache_path: str
    likely_coder: bool


class CodingAgentBackend(Protocol):
    """Backend that chooses how to repair one RealRepoFix task."""

    name: str
    is_full_llm: bool

    def repair(self, task: RealRepoFixTask) -> AgentTrajectory:
        """Run read/search/edit/test and return a trajectory."""


def discover_local_models(
    cache_root: Path = Path.home() / ".cache" / "huggingface" / "hub",
) -> list[LocalModelCandidate]:
    """Discover locally cached HF model directories without network access."""

    if not cache_root.exists():
        return []

    candidates: list[LocalModelCandidate] = []
    for path in sorted(cache_root.glob("models--*")):
        model_id = path.name.removeprefix("models--").replace("--", "/")
        lowered = model_id.lower()
        likely_coder = any(
            marker in lowered
            for marker in ["coder", "code", "deepseek", "qwen"]
        )
        candidates.append(
            LocalModelCandidate(
                model_id=model_id,
                cache_path=str(path),
                likely_coder=likely_coder,
            )
        )

    return candidates


def extract_replacement_source(generated_text: str) -> str | None:
    """Extract a valid Python replacement file from model output."""

    candidates = [
        match.group(1).strip()
        for match in PYTHON_FENCE_RE.finditer(generated_text)
    ]
    stripped = generated_text.strip()
    if stripped:
        candidates.append(stripped)

    for candidate in candidates:
        if not candidate or "def " not in candidate:
            continue
        try:
            ast.parse(candidate)
        except SyntaxError:
            continue
        return candidate if candidate.endswith("\n") else candidate + "\n"

    return None


def search_task_context(task: RealRepoFixTask) -> str:
    """Return a compact real-repo search result for the task."""

    test_path = task.repo_dir / task.test_file
    if not test_path.exists():
        return f"{task.test_file}: <missing>"
    test_source = test_path.read_text(encoding="utf-8")
    return f"{task.test_file}\n```python\n{test_source}\n```"


@dataclass
class HuggingFaceRepairBackend:
    """Repair RealRepoFix tasks with a local or downloadable HF coder model."""

    model_path: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    adapter_path: Path | None = None
    max_new_tokens: int = 384
    temperature: float = 0.0
    local_files_only: bool = False
    name: str = "qwen2.5-coder-7b"
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

    def _prompt(
        self,
        task: RealRepoFixTask,
        source_before: str,
        search_result: str,
        pytest_before: str,
    ) -> str:
        return (
            "You are a coding agent repairing a small Python repository.\n"
            "Return only the full replacement contents for the source file, "
            "preferably in one ```python fenced block. Do not include prose.\n\n"
            f"Issue:\n{task.issue}\n\n"
            f"Source file: {task.source_file}\n"
            f"Current contents:\n```python\n{source_before}\n```\n\n"
            f"Search result:\n{search_result}\n\n"
            f"Failing pytest output:\n```\n{pytest_before[-4000:]}\n```\n"
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

    def repair(self, task: RealRepoFixTask) -> AgentTrajectory:
        source_path = task.repo_dir / task.source_file
        source_path.write_text(task.buggy_source, encoding="utf-8")
        source_before = source_path.read_text(encoding="utf-8")
        search_result = search_task_context(task)
        before = run_pytest_reward(task)
        prompt = self._prompt(
            task,
            source_before,
            search_result,
            f"returncode={before.returncode}\n{before.stdout}\n{before.stderr}",
        )
        generated_text, generation_metadata = self._generate(prompt)
        replacement = extract_replacement_source(generated_text)
        patch_applied = replacement is not None
        if replacement is not None:
            source_path.write_text(replacement, encoding="utf-8")
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
                tool_name="search",
                content=f"search relevant tests for {task.source_file}",
                cost=1.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="search",
                content=search_result,
                cost=0.5,
                metadata={"test_file": str(task.test_file)},
            ),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="edit",
                content=f"replace {task.source_file} with generated repair",
                cost=2.0,
                metadata=generation_metadata,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="edit",
                content=replacement if replacement is not None else generated_text,
                cost=0.5,
                metadata={
                    "patch_applied": patch_applied,
                    "backend": self.name,
                    "model_path": self.model_path,
                    "raw_generated_text": generated_text,
                },
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
                content=f"pytest returncode={after.returncode}\n{after.stdout}\n{after.stderr}",
                cost=1.0,
                metadata={"pytest_reward": after.reward},
            ),
        )
        return AgentTrajectory(
            trajectory_id=f"{task.task_id}-{self.name}",
            events=events,
            final_reward=after.reward,
            metadata={
                "benchmark": "realrepofix",
                "template": task.template,
                "repo_dir": str(task.repo_dir),
                "agent_backend": self.name,
                "full_llm": self.is_full_llm,
                "model_path": self.model_path,
                "patch_applied": patch_applied,
            },
        )


class ReActRetryRepairBackend(HuggingFaceRepairBackend):
    """HF repair backend that retries once after a failed edit/test attempt."""

    name: str = "react_retry_test"
    max_attempts: int = 2

    def _retry_prompt(
        self,
        task: RealRepoFixTask,
        source_before: str,
        search_result: str,
        pytest_before: str,
        previous_output: str,
        previous_test_output: str,
    ) -> str:
        return self._prompt(task, source_before, search_result, pytest_before) + (
            "\nThe previous attempt failed. Try one more repair.\n"
            "Previous model output:\n"
            f"```\n{previous_output[-2000:]}\n```\n"
            "Previous pytest output:\n"
            f"```\n{previous_test_output[-4000:]}\n```\n"
        )

    def repair(self, task: RealRepoFixTask) -> AgentTrajectory:
        source_path = task.repo_dir / task.source_file
        source_path.write_text(task.buggy_source, encoding="utf-8")
        source_before = source_path.read_text(encoding="utf-8")
        search_result = search_task_context(task)
        before = run_pytest_reward(task)
        pytest_before = f"returncode={before.returncode}\n{before.stdout}\n{before.stderr}"

        events: list[AgentEvent] = [
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
                tool_name="search",
                content=f"search relevant tests for {task.source_file}",
                cost=1.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="search",
                content=search_result,
                cost=0.5,
                metadata={"test_file": str(task.test_file)},
            ),
        ]

        final_reward = 0.0
        patch_applied = False
        previous_output = ""
        previous_test_output = pytest_before
        for attempt_index in range(self.max_attempts):
            if attempt_index == 0:
                prompt = self._prompt(task, source_before, search_result, pytest_before)
            else:
                source_path.write_text(task.buggy_source, encoding="utf-8")
                prompt = self._retry_prompt(
                    task,
                    source_before,
                    search_result,
                    pytest_before,
                    previous_output,
                    previous_test_output,
                )
            generated_text, generation_metadata = self._generate(prompt)
            replacement = extract_replacement_source(generated_text)
            attempt_patch_applied = replacement is not None
            if replacement is not None:
                source_path.write_text(replacement, encoding="utf-8")
            after = run_pytest_reward(task)
            final_reward = after.reward
            patch_applied = patch_applied or attempt_patch_applied
            previous_output = generated_text
            previous_test_output = (
                f"pytest returncode={after.returncode}\n{after.stdout}\n{after.stderr}"
            )
            events.extend(
                [
                    AgentEvent(
                        kind=TOOL_CALL,
                        tool_name="edit",
                        content=f"attempt {attempt_index + 1}: replace {task.source_file}",
                        cost=2.0,
                        metadata={**generation_metadata, "attempt": attempt_index + 1},
                    ),
                    AgentEvent(
                        kind=TOOL_RESULT,
                        tool_name="edit",
                        content=replacement if replacement is not None else generated_text,
                        cost=0.5,
                        metadata={
                            "patch_applied": attempt_patch_applied,
                            "backend": self.name,
                            "model_path": self.model_path,
                            "raw_generated_text": generated_text,
                            "attempt": attempt_index + 1,
                        },
                    ),
                    AgentEvent(
                        kind=TOOL_CALL,
                        tool_name="run_test",
                        content=" ".join(task.test_command),
                        cost=2.0,
                        metadata={"attempt": attempt_index + 1},
                    ),
                    AgentEvent(
                        kind=TOOL_RESULT,
                        tool_name="run_test",
                        content=previous_test_output,
                        cost=1.0,
                        metadata={
                            "pytest_reward": after.reward,
                            "attempt": attempt_index + 1,
                        },
                    ),
                ]
            )
            if after.reward > 0.0:
                break

        return AgentTrajectory(
            trajectory_id=f"{task.task_id}-{self.name}",
            events=tuple(events),
            final_reward=final_reward,
            metadata={
                "benchmark": "realrepofix",
                "template": task.template,
                "repo_dir": str(task.repo_dir),
                "agent_backend": self.name,
                "full_llm": self.is_full_llm,
                "model_path": self.model_path,
                "patch_applied": patch_applied,
            },
        )


class ReflexionRepairBackend(ReActRetryRepairBackend):
    """Retry backend that explicitly frames the second attempt as reflection."""

    name: str = "reflexion_self_repair"

    def _retry_prompt(
        self,
        task: RealRepoFixTask,
        source_before: str,
        search_result: str,
        pytest_before: str,
        previous_output: str,
        previous_test_output: str,
    ) -> str:
        return self._prompt(task, source_before, search_result, pytest_before) + (
            "\nPrevious attempt failed. Reflect on the pytest failure and return "
            "only a corrected full replacement file.\n"
            "Previous model output:\n"
            f"```\n{previous_output[-2000:]}\n```\n"
            "Failure feedback:\n"
            f"```\n{previous_test_output[-4000:]}\n```\n"
        )


@dataclass(frozen=True)
class ScriptedRepairBackend:
    """Deterministic backend that applies the known expected fix."""

    name: str = "scripted"
    is_full_llm: bool = False

    def repair(self, task: RealRepoFixTask) -> AgentTrajectory:
        source_path = task.repo_dir / task.source_file
        source_path.write_text(task.buggy_source, encoding="utf-8")
        source_before = source_path.read_text(encoding="utf-8")
        search_result = search_task_context(task)
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
                tool_name="search",
                content=f"search relevant tests for {task.source_file}",
                cost=1.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="search",
                content=search_result,
                cost=0.5,
                metadata={"test_file": str(task.test_file)},
            ),
            AgentEvent(
                kind=TOOL_CALL,
                tool_name="edit",
                content=f"scripted repair for {task.source_file}",
                cost=2.0,
            ),
            AgentEvent(
                kind=TOOL_RESULT,
                tool_name="edit",
                content=task.fixed_source,
                cost=0.5,
                metadata={"patch_applied": True, "backend": self.name},
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
        return AgentTrajectory(
            trajectory_id=f"{task.task_id}-{self.name}",
            events=events,
            final_reward=after.reward,
            metadata={
                "benchmark": "realrepofix",
                "template": task.template,
                "repo_dir": str(task.repo_dir),
                "agent_backend": self.name,
                "full_llm": self.is_full_llm,
            },
        )
