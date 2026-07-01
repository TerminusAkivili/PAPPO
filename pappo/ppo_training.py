"""PPO utilities for PAPPO turn-level policy optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, pstdev

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from pappo.lora_training import load_model_and_tokenizer, lora_target_modules


@dataclass(frozen=True)
class PPOTurnSample:
    """One action-only tool-turn sample for PPO training."""

    trajectory_id: str
    turn_id: int
    tool_name: str
    prompt: str
    action_text: str
    target: float
    value: float
    old_logprobs: tuple[float, ...]
    ref_logprobs: tuple[float, ...]
    action_mask: tuple[int, ...]


@dataclass(frozen=True)
class PPOLossMetrics:
    """Differentiable PPO loss tensors plus scalar diagnostics."""

    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl_loss: torch.Tensor
    value_loss: torch.Tensor
    ratio_mean: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: float


@dataclass(frozen=True)
class PPOTrainResult:
    """Summary from one PAPPO-PPO LoRA update."""

    status: str
    model: str
    samples: int
    ppo_epochs: int
    optimizer_steps: int
    normalize_advantages: bool
    trainable_parameters: int
    loss: float
    policy_loss: float
    kl: float
    value_loss: float
    ratio_mean: float
    clip_fraction: float
    adapter_dir: str


def normalize_advantages(values: list[float], epsilon: float = 1e-8) -> list[float]:
    """Normalize scalar advantages across one PPO update batch."""

    if not values:
        return []
    center = mean(values)
    scale = pstdev(values) or 1.0
    return [(value - center) / (scale + epsilon) for value in values]


def prepare_sample_advantages(
    samples: list[PPOTurnSample],
    *,
    normalize: bool,
) -> list[float]:
    """Compute scalar sample advantages with optional batch normalization."""

    raw_advantages = [sample.target - sample.value for sample in samples]
    if (
        normalize
        and len(raw_advantages) > 1
        and (pstdev(raw_advantages) or 0.0) > 0.0
    ):
        return normalize_advantages(raw_advantages)
    return raw_advantages


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denominator = mask.sum().clamp_min(1.0)
    return (values * mask).sum() / denominator


def compute_turn_ppo_loss(
    *,
    new_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    action_mask: torch.Tensor,
    advantage: torch.Tensor,
    value: torch.Tensor,
    target: torch.Tensor,
    clip_epsilon: float,
    kl_beta: float,
    value_coef: float,
) -> PPOLossMetrics:
    """Compute one turn-level clipped PPO loss over action tokens only."""

    ratio = torch.exp(new_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    unclipped_objective = ratio * advantage
    clipped_objective = clipped_ratio * advantage
    policy_loss = -_masked_mean(
        torch.minimum(unclipped_objective, clipped_objective),
        action_mask,
    )
    log_ratio_to_ref = new_logprobs - ref_logprobs
    token_kl = torch.exp(log_ratio_to_ref) - 1.0 - log_ratio_to_ref
    approx_kl = _masked_mean(token_kl, action_mask)
    kl_loss = kl_beta * approx_kl
    value_loss = torch.nn.functional.mse_loss(value, target)
    loss = policy_loss + kl_loss + value_coef * value_loss
    clipped_tokens = ((ratio - clipped_ratio).abs() > 1e-8).float()
    clip_fraction = float(_masked_mean(clipped_tokens, action_mask).detach().cpu())
    return PPOLossMetrics(
        loss=loss,
        policy_loss=policy_loss,
        kl_loss=kl_loss,
        value_loss=value_loss,
        ratio_mean=_masked_mean(ratio, action_mask),
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
    )


def gather_response_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Gather next-token logprobs and zero non-response positions."""

    log_probs = torch.nn.functional.log_softmax(logits[:, :-1, :], dim=-1)
    next_tokens = input_ids[:, 1:].unsqueeze(-1)
    gathered = log_probs.gather(dim=-1, index=next_tokens).squeeze(-1)
    output = torch.zeros_like(input_ids, dtype=logits.dtype)
    output[:, 1:] = gathered * response_mask[:, 1:].to(logits.dtype)
    return output


def make_sample_full_text(sample: PPOTurnSample) -> str:
    """Join prompt and action without changing exact generation prompt suffixes."""

    separator = "" if sample.prompt.endswith((" ", "\t", "\n")) else "\n"
    return f"{sample.prompt}{separator}{sample.action_text}"


def _sample_tensors(tokenizer, sample: PPOTurnSample, max_length: int, device):
    prompt_ids = tokenizer(
        sample.prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    full_text = make_sample_full_text(sample)
    encoded = tokenizer(
        full_text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    response_mask = torch.zeros_like(input_ids, dtype=torch.float32, device=device)
    response_start = min(len(prompt_ids), input_ids.shape[1])
    response_mask[:, response_start:] = attention_mask[:, response_start:].float()
    return input_ids, attention_mask, response_mask


def _align_sample_vector(values: tuple[float, ...], length: int, device) -> torch.Tensor:
    if not values:
        raise ValueError("PPO sample is missing stored logprobs")
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if tensor.numel() >= length:
        return tensor[:length]
    padding = torch.full(
        (length - tensor.numel(),),
        float(tensor[-1]),
        dtype=torch.float32,
        device=device,
    )
    return torch.cat([tensor, padding], dim=0)


def _align_sample_mask(values: tuple[int, ...], length: int, device) -> torch.Tensor:
    if not values:
        raise ValueError("PPO sample is missing an action mask")
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if tensor.numel() >= length:
        return tensor[:length]
    padding = torch.zeros(length - tensor.numel(), dtype=torch.float32, device=device)
    return torch.cat([tensor, padding], dim=0)


def _load_policy_with_optional_adapter(
    model_name: str,
    *,
    adapter_path: Path | None,
    local_files_only: bool,
    trainable_adapter: bool,
):
    device_map = {"": 0} if torch.cuda.is_available() else "auto"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, tokenizer, model_label = load_model_and_tokenizer(
        model_name,
        local_files_only=local_files_only,
        device_map=device_map,
    )
    if adapter_path is not None:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=trainable_adapter,
            device_map=device_map,
        )
    return model, tokenizer, model_label


def _compute_sample_logprobs_for_policy(
    *,
    model_name: str,
    samples: list[PPOTurnSample],
    adapter_path: Path | None,
    max_length: int,
    local_files_only: bool,
) -> tuple[list[tuple[float, ...]], list[tuple[int, ...]]]:
    model, tokenizer, _model_label = _load_policy_with_optional_adapter(
        model_name,
        adapter_path=adapter_path,
        local_files_only=local_files_only,
        trainable_adapter=False,
    )
    model.eval()
    device = next(model.parameters()).device
    all_logprobs: list[tuple[float, ...]] = []
    all_masks: list[tuple[int, ...]] = []
    with torch.no_grad():
        for sample in samples:
            input_ids, attention_mask, response_mask = _sample_tensors(
                tokenizer, sample, max_length, device
            )
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logprobs = gather_response_logprobs(
                output.logits, input_ids, response_mask
            )[0]
            all_logprobs.append(
                tuple(float(value) for value in logprobs.detach().float().cpu())
            )
            all_masks.append(
                tuple(int(value) for value in response_mask[0].detach().int().cpu())
            )
            del input_ids, attention_mask, response_mask, output, logprobs
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return all_logprobs, all_masks


def fill_samples_with_model_logprobs(
    *,
    model_name: str,
    samples: list[PPOTurnSample],
    old_adapter_path: Path | None,
    max_length: int = 384,
    local_files_only: bool = False,
) -> list[PPOTurnSample]:
    """Attach old-policy and reference-policy logprobs to PPO samples."""

    if not samples:
        return []
    old_logprobs, action_masks = _compute_sample_logprobs_for_policy(
        model_name=model_name,
        samples=samples,
        adapter_path=old_adapter_path,
        max_length=max_length,
        local_files_only=local_files_only,
    )
    if old_adapter_path is None:
        ref_logprobs = old_logprobs
    else:
        ref_logprobs, _ref_masks = _compute_sample_logprobs_for_policy(
            model_name=model_name,
            samples=samples,
            adapter_path=None,
            max_length=max_length,
            local_files_only=local_files_only,
        )
    return [
        replace(
            sample,
            old_logprobs=old,
            ref_logprobs=ref,
            action_mask=mask,
        )
        for sample, old, ref, mask in zip(
            samples,
            old_logprobs,
            ref_logprobs,
            action_masks,
            strict=True,
        )
    ]


def apply_advantage_prior(
    samples: list[PPOTurnSample],
    *,
    prior: float,
) -> list[PPOTurnSample]:
    """Fold a PAPPO-v2 local prior into sample values for advantage computation."""

    if prior == 0.0:
        return samples
    return [
        replace(sample, value=float(sample.value) - float(prior) * float(sample.target))
        for sample in samples
    ]


def train_pappo_ppo_update(
    *,
    model_name: str,
    samples: list[PPOTurnSample],
    output_dir: Path,
    input_adapter_path: Path | None = None,
    max_length: int = 384,
    learning_rate: float = 1e-5,
    ppo_epochs: int = 1,
    normalize_advantages_flag: bool = True,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.01,
    value_coef: float = 0.1,
    local_files_only: bool = False,
) -> PPOTrainResult:
    """Run one action-masked LoRA PPO update over saved turn samples."""

    if not samples:
        raise ValueError("no PPO samples found")
    if ppo_epochs < 1:
        raise ValueError("ppo_epochs must be >= 1")
    model, tokenizer, model_label = _load_policy_with_optional_adapter(
        model_name,
        adapter_path=input_adapter_path,
        local_files_only=local_files_only,
        trainable_adapter=True,
    )
    if input_adapter_path is None:
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
    advantages = prepare_sample_advantages(
        samples,
        normalize=normalize_advantages_flag,
    )

    loss_total = 0.0
    policy_total = 0.0
    kl_total = 0.0
    value_total = 0.0
    ratio_total = 0.0
    clip_total = 0.0
    normalizer = float(len(samples) * ppo_epochs)
    for _epoch in range(ppo_epochs):
        optimizer.zero_grad(set_to_none=True)
        for sample, advantage in zip(samples, advantages, strict=True):
            input_ids, attention_mask, response_mask = _sample_tensors(
                tokenizer, sample, max_length, device
            )
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            new_logprobs = gather_response_logprobs(
                output.logits, input_ids, response_mask
            )[0]
            old_logprobs = _align_sample_vector(
                sample.old_logprobs,
                new_logprobs.numel(),
                device,
            )
            ref_logprobs = _align_sample_vector(
                sample.ref_logprobs,
                new_logprobs.numel(),
                device,
            )
            action_mask = _align_sample_mask(
                sample.action_mask,
                new_logprobs.numel(),
                device,
            )
            metrics = compute_turn_ppo_loss(
                new_logprobs=new_logprobs,
                old_logprobs=old_logprobs,
                ref_logprobs=ref_logprobs,
                action_mask=action_mask,
                advantage=torch.tensor(float(advantage), device=device),
                value=torch.tensor(float(sample.value), device=device),
                target=torch.tensor(float(sample.target), device=device),
                clip_epsilon=clip_epsilon,
                kl_beta=kl_beta,
                value_coef=value_coef,
            )
            loss_total += float(metrics.loss.detach().cpu())
            policy_total += float(metrics.policy_loss.detach().cpu())
            kl_total += float(metrics.approx_kl.detach().cpu())
            value_total += float(metrics.value_loss.detach().cpu())
            ratio_total += float(metrics.ratio_mean.detach().cpu())
            clip_total += metrics.clip_fraction
            (metrics.loss / float(len(samples))).backward()
            del input_ids, attention_mask, response_mask, output
        optimizer.step()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    result = PPOTrainResult(
        status="pappo_ppo_update_completed",
        model=model_label,
        samples=len(samples),
        ppo_epochs=ppo_epochs,
        optimizer_steps=ppo_epochs,
        normalize_advantages=normalize_advantages_flag,
        trainable_parameters=trainable_parameters,
        loss=round(loss_total / normalizer, 6),
        policy_loss=round(policy_total / normalizer, 6),
        kl=round(kl_total / normalizer, 6),
        value_loss=round(value_total / normalizer, 6),
        ratio_mean=round(ratio_total / normalizer, 6),
        clip_fraction=round(clip_total / normalizer, 6),
        adapter_dir=str(output_dir),
    )
    del model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
