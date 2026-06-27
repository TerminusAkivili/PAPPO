"""LoRA training helpers for PAPPO tool-call turns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from pappo.trajectory import AgentTrajectory, split_tool_call_turns
from pappo.trl_adapter import turn_response_text


LORA_METHODS = (
    "trace",
    "token_broadcast",
    "pappo_turn",
    "pappo_turn_v2",
    "pappo_turn_v3",
    "grpo_lite",
)
PAPPO_V2_LOCAL_PRIOR = 0.25


def turn_local_target(tool_name: str, result_metadata: dict, final_reward: float) -> float:
    """Return a local turn target from tool result metadata when available."""

    if "pytest_reward" in result_metadata:
        return float(result_metadata["pytest_reward"])
    if "patch_applied" in result_metadata:
        return 1.0 if result_metadata["patch_applied"] else 0.0
    if "progress" in result_metadata:
        return float(result_metadata["progress"])
    if "pytest_before_reward" in result_metadata:
        return float(result_metadata["pytest_before_reward"])
    if tool_name == "run_test":
        return final_reward
    return 0.5 * final_reward


@dataclass(frozen=True)
class TrainingExample:
    """One weighted language-model training example."""

    prompt: str
    response: str
    weight: float


@dataclass(frozen=True)
class LoraTrainResult:
    """Result from one LoRA pilot update."""

    status: str
    model: str
    method: str
    examples: int
    epochs: int
    trainable_parameters: int
    loss: float
    adapter_dir: str
    checkpoints: int = 0


def make_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Create a tiny tokenizer for tests and CPU-fast smoke runs."""

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "<task>": 4,
        "<tool_call": 5,
        "name": 6,
        "read_file": 7,
        "search": 8,
        "edit": 9,
        "run_test": 10,
        "</tool_call>": 11,
        "<tool_result>": 12,
        "</tool_result>": 13,
        "def": 14,
        "return": 15,
        "pytest": 16,
        "passed": 17,
        "failed": 18,
        "=": 19,
    }
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
    )


def make_tiny_model(vocab_size: int) -> GPT2LMHeadModel:
    """Create a tiny causal LM for fast LoRA tests."""

    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=512,
        n_ctx=512,
        n_embd=32,
        n_layer=1,
        n_head=2,
        bos_token_id=2,
        eos_token_id=3,
        pad_token_id=0,
    )
    return GPT2LMHeadModel(config)


def load_model_and_tokenizer(
    model_name: str,
    local_files_only: bool,
) -> tuple[torch.nn.Module, PreTrainedTokenizerFast | AutoTokenizer, str]:
    """Load the tiny local model or a Hugging Face causal LM."""

    if model_name == "tiny-local":
        tokenizer = make_tiny_tokenizer()
        return make_tiny_model(len(tokenizer)), tokenizer, "tiny-local"

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    return model, tokenizer, model_name


def lora_target_modules(model_name: str) -> list[str]:
    """Return LoRA target module names for supported model families."""

    if model_name == "tiny-local":
        return ["c_attn"]
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def trajectory_task_group(trajectory: AgentTrajectory) -> str:
    """Return the most local task group available for counterfactual baselines."""

    return str(
        trajectory.metadata.get("task_id")
        or trajectory.metadata.get("repo_dir")
        or trajectory.metadata.get("template")
        or trajectory.trajectory_id
    )


def pappo_turn_v2_score(trajectory: AgentTrajectory, turn) -> float:
    """Score a turn with local, outcome-aware credit for PAPPO-v2."""

    result_metadata = dict(turn.metadata.get("result_metadata", {}))
    final_reward = float(trajectory.final_reward)
    if turn.tool_name == "edit":
        patch_applied = bool(result_metadata.get("patch_applied", False))
        if patch_applied and final_reward > 0.0:
            return 1.0
        if patch_applied:
            return -1.0
        return -0.5
    if "pytest_reward" in result_metadata:
        return 1.0 if float(result_metadata["pytest_reward"]) > 0.0 else -1.0
    if turn.tool_name == "run_test":
        return 1.0 if final_reward > 0.0 else -1.0
    if "pytest_before_reward" in result_metadata:
        return 0.0
    if turn.tool_name == "search":
        useful = "tests/" in turn.tool_result and "assert " in turn.tool_result
        return 0.25 if useful else -0.25
    return 0.5 if final_reward > 0.0 else -0.5


def _pappo_v2_baselines(
    trajectories: list[AgentTrajectory],
) -> dict[tuple[str, int, str], float]:
    grouped_scores: dict[tuple[str, int, str], list[float]] = {}
    for trajectory in trajectories:
        group = trajectory_task_group(trajectory)
        for turn in split_tool_call_turns(trajectory):
            key = (group, turn.turn_id, turn.tool_name)
            grouped_scores.setdefault(key, []).append(pappo_turn_v2_score(trajectory, turn))
    return {
        key: float(mean(scores))
        for key, scores in grouped_scores.items()
        if len(scores) > 1
    }


def build_training_examples(
    trajectories: list[AgentTrajectory],
    method: str,
    limit_examples: int | None = None,
) -> list[TrainingExample]:
    """Build weighted examples for one comparison method."""

    if method not in LORA_METHODS:
        raise ValueError(f"unknown LoRA comparison method: {method}")

    group_means: dict[str, float] = {}
    if method == "grpo_lite":
        group_rewards: dict[str, list[float]] = {}
        for trajectory in trajectories:
            group = str(trajectory.metadata.get("template", "unknown"))
            group_rewards.setdefault(group, []).append(float(trajectory.final_reward))
        group_means = {
            group: float(mean(rewards))
            for group, rewards in group_rewards.items()
        }
    pappo_v2_baselines = (
        _pappo_v2_baselines(trajectories)
        if method in {"pappo_turn_v2", "pappo_turn_v3"}
        else {}
    )

    examples: list[TrainingExample] = []
    for trajectory in trajectories:
        turns = split_tool_call_turns(trajectory)
        if not turns:
            continue
        if method == "trace":
            examples.append(
                TrainingExample(
                    prompt=turns[0].prompt or "<task>",
                    response="\n".join(turn_response_text(turn) for turn in turns),
                    weight=float(trajectory.final_reward),
                )
            )
        else:
            group = str(trajectory.metadata.get("template", "unknown"))
            for turn in turns:
                if method == "token_broadcast":
                    weight = float(trajectory.final_reward)
                elif method == "pappo_turn":
                    weight = turn_local_target(
                        turn.tool_name,
                        dict(turn.metadata.get("result_metadata", {})),
                        float(trajectory.final_reward),
                    )
                elif method == "pappo_turn_v2":
                    key = (trajectory_task_group(trajectory), turn.turn_id, turn.tool_name)
                    score = pappo_turn_v2_score(trajectory, turn)
                    weight = (
                        score
                        - pappo_v2_baselines.get(key, 0.0)
                        + PAPPO_V2_LOCAL_PRIOR * score
                    )
                elif method == "pappo_turn_v3":
                    key = (trajectory_task_group(trajectory), turn.turn_id, turn.tool_name)
                    score = pappo_turn_v2_score(trajectory, turn)
                    counterfactual = score - pappo_v2_baselines.get(key, 0.0)
                    weight = (
                        float(trajectory.final_reward)
                        + 0.5 * counterfactual
                        + PAPPO_V2_LOCAL_PRIOR * score
                    )
                else:
                    weight = float(trajectory.final_reward) - group_means.get(group, 0.0)
                examples.append(
                    TrainingExample(
                        prompt=turn.prompt or "<task>",
                        response=turn_response_text(turn),
                        weight=weight,
                    )
                )
        if limit_examples is not None and len(examples) >= limit_examples:
            return examples[:limit_examples]

    return examples


def _example_tensors(
    tokenizer,
    example: TrainingExample,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    full_text = f"{example.prompt}\n{example.response}"
    prompt_ids = tokenizer(
        example.prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[:, : min(len(prompt_ids), labels.shape[1])] = -100
    return input_ids, attention_mask, labels


def train_lora_adapter(
    *,
    model_name: str,
    trajectories: list[AgentTrajectory],
    method: str,
    output_dir: Path,
    limit_examples: int | None = None,
    max_length: int = 384,
    learning_rate: float = 1e-4,
    epochs: int = 1,
    local_files_only: bool = False,
    checkpoint_dirs_by_example: dict[int, Path] | None = None,
) -> LoraTrainResult:
    """Run one weighted LoRA update and save the adapter."""

    examples = build_training_examples(trajectories, method, limit_examples)
    if not examples:
        raise ValueError("no training examples found")
    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    model, tokenizer, model_label = load_model_and_tokenizer(
        model_name,
        local_files_only=local_files_only,
    )
    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=lora_target_modules(model_name),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.train()
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )

    checkpoint_dirs_by_example = checkpoint_dirs_by_example or {}
    checkpoint_points = set(checkpoint_dirs_by_example)
    saved_checkpoints: set[int] = set()
    loss_total = 0.0
    normalizer = float(len(examples))
    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        for example_index, example in enumerate(examples, start=1):
            input_ids, attention_mask, labels = _example_tensors(
                tokenizer,
                example,
                max_length=max_length,
                device=device,
            )
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            weighted_loss = output.loss * float(example.weight)
            loss_total += float(weighted_loss.detach().cpu())
            (weighted_loss / normalizer).backward()
            del input_ids, attention_mask, labels, output, weighted_loss
            if _epoch == 0 and example_index in checkpoint_points:
                optimizer.step()
                checkpoint_dir = checkpoint_dirs_by_example[example_index]
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                saved_checkpoints.add(example_index)
                optimizer.zero_grad(set_to_none=True)
        optimizer.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    result = LoraTrainResult(
        status="lora_pappo_step_completed",
        model=model_label,
        method=method,
        examples=len(examples),
        epochs=epochs,
        trainable_parameters=trainable_parameters,
        loss=round(loss_total / (normalizer * epochs), 6),
        adapter_dir=str(output_dir),
        checkpoints=len(saved_checkpoints),
    )
    del model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
