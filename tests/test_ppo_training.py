from __future__ import annotations

import torch
import pytest

from pappo.ppo_training import (
    PPOTurnSample,
    apply_advantage_prior,
    compute_turn_ppo_loss,
    fill_samples_with_model_logprobs,
    make_sample_full_text,
    normalize_advantages,
    prepare_sample_advantages,
    train_pappo_ppo_update,
)


def test_normalize_advantages_centers_and_scales() -> None:
    normalized = normalize_advantages([1.0, 2.0, 3.0])

    assert round(sum(normalized), 6) == 0.0
    assert normalized[0] < 0.0
    assert normalized[2] > 0.0


def test_prepare_sample_advantages_can_skip_normalization() -> None:
    samples = [
        PPOTurnSample(
            trajectory_id="t1",
            turn_id=0,
            tool_name="edit",
            prompt="<task>",
            action_text="good",
            target=1.0,
            value=0.75,
            old_logprobs=(-0.7,),
            ref_logprobs=(-0.7,),
            action_mask=(1,),
        ),
        PPOTurnSample(
            trajectory_id="t2",
            turn_id=0,
            tool_name="edit",
            prompt="<task>",
            action_text="bad",
            target=-1.0,
            value=-0.75,
            old_logprobs=(-0.7,),
            ref_logprobs=(-0.7,),
            action_mask=(1,),
        ),
    ]

    assert prepare_sample_advantages(samples, normalize=False) == [0.25, -0.25]
    normalized = prepare_sample_advantages(samples, normalize=True)
    assert normalized[0] == pytest.approx(1.0)
    assert normalized[1] == pytest.approx(-1.0)


def test_compute_turn_ppo_loss_clips_positive_advantage_ratio() -> None:
    old_logprobs = torch.log(torch.tensor([0.5, 0.5]))
    new_logprobs = torch.log(torch.tensor([0.9, 0.9]))
    ref_logprobs = torch.log(torch.tensor([0.5, 0.5]))
    mask = torch.tensor([1.0, 1.0])

    metrics = compute_turn_ppo_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        action_mask=mask,
        advantage=torch.tensor(1.0),
        value=torch.tensor(0.0),
        target=torch.tensor(1.0),
        clip_epsilon=0.2,
        kl_beta=0.0,
        value_coef=0.0,
    )

    assert metrics.clip_fraction == 1.0
    assert torch.isclose(metrics.policy_loss, torch.tensor(-1.2), atol=1e-5)


def test_compute_turn_ppo_loss_ignores_masked_environment_tokens() -> None:
    old_logprobs = torch.log(torch.tensor([0.5, 0.5]))
    new_logprobs = torch.log(torch.tensor([0.9, 0.1]))
    ref_logprobs = torch.log(torch.tensor([0.5, 0.5]))
    mask = torch.tensor([1.0, 0.0])

    metrics = compute_turn_ppo_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        action_mask=mask,
        advantage=torch.tensor(1.0),
        value=torch.tensor(0.0),
        target=torch.tensor(1.0),
        clip_epsilon=0.2,
        kl_beta=0.0,
        value_coef=0.0,
    )

    assert torch.isclose(metrics.ratio_mean, torch.tensor(1.8), atol=1e-5)


def test_compute_turn_ppo_loss_kl_has_zero_gradient_at_reference_policy() -> None:
    new_logprobs = torch.tensor([-0.7, -0.7], requires_grad=True)
    old_logprobs = torch.tensor([-0.7, -0.7])
    ref_logprobs = torch.tensor([-0.7, -0.7])
    mask = torch.tensor([1.0, 1.0])

    metrics = compute_turn_ppo_loss(
        new_logprobs=new_logprobs,
        old_logprobs=old_logprobs,
        ref_logprobs=ref_logprobs,
        action_mask=mask,
        advantage=torch.tensor(0.0),
        value=torch.tensor(0.0),
        target=torch.tensor(0.0),
        clip_epsilon=0.2,
        kl_beta=1.0,
        value_coef=0.0,
    )
    metrics.loss.backward()

    assert torch.isclose(metrics.approx_kl, torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(new_logprobs.grad, torch.zeros_like(new_logprobs), atol=1e-6)


def test_gather_response_logprobs_uses_shifted_labels() -> None:
    logits = torch.zeros(1, 3, 2)
    logits[0, 0, 1] = 4.0
    logits[0, 1, 0] = 4.0
    input_ids = torch.tensor([[0, 1, 0]])
    mask = torch.tensor([[0.0, 1.0, 1.0]])

    from pappo.ppo_training import gather_response_logprobs

    logprobs = gather_response_logprobs(logits, input_ids, mask)

    assert logprobs.shape == input_ids.shape
    assert logprobs[0, 0].item() == 0.0
    assert logprobs[0, 1].item() > -0.1


def test_make_sample_full_text_preserves_exact_prompt_suffix() -> None:
    sample = PPOTurnSample(
        trajectory_id="t1",
        turn_id=0,
        tool_name="edit",
        prompt="<chat>assistant\n",
        action_text="def fixed(): return True",
        target=1.0,
        value=0.0,
        old_logprobs=(-0.7,),
        ref_logprobs=(-0.7,),
        action_mask=(1,),
    )

    assert make_sample_full_text(sample) == "<chat>assistant\ndef fixed(): return True"


def test_train_pappo_ppo_update_saves_adapter_and_metrics(tmp_path) -> None:
    sample = PPOTurnSample(
        trajectory_id="t1",
        turn_id=0,
        tool_name="edit",
        prompt="<task>",
        action_text="def return",
        target=1.0,
        value=0.0,
        old_logprobs=(-0.7, -0.7, -0.7),
        ref_logprobs=(-0.7, -0.7, -0.7),
        action_mask=(1, 1, 1),
    )

    result = train_pappo_ppo_update(
        model_name="tiny-local",
        samples=[sample],
        output_dir=tmp_path / "adapter",
        local_files_only=True,
    )

    assert result.status == "pappo_ppo_update_completed"
    assert result.samples == 1
    assert result.policy_loss != 0.0
    assert (tmp_path / "adapter" / "adapter_config.json").exists()


def test_train_pappo_ppo_update_runs_multiple_ppo_epochs(tmp_path) -> None:
    sample = PPOTurnSample(
        trajectory_id="t1",
        turn_id=0,
        tool_name="edit",
        prompt="<task>",
        action_text="def return",
        target=1.0,
        value=0.0,
        old_logprobs=(-0.7, -0.7, -0.7),
        ref_logprobs=(-0.7, -0.7, -0.7),
        action_mask=(1, 1, 1),
    )

    result = train_pappo_ppo_update(
        model_name="tiny-local",
        samples=[sample],
        output_dir=tmp_path / "adapter",
        ppo_epochs=2,
        local_files_only=True,
    )

    assert result.ppo_epochs == 2
    assert result.optimizer_steps == 2
    assert result.ratio_mean > 0.0


def test_train_pappo_ppo_update_can_continue_from_existing_adapter(tmp_path) -> None:
    sample = PPOTurnSample(
        trajectory_id="t1",
        turn_id=0,
        tool_name="edit",
        prompt="<task>",
        action_text="def return",
        target=1.0,
        value=0.0,
        old_logprobs=(-0.7, -0.7, -0.7),
        ref_logprobs=(-0.7, -0.7, -0.7),
        action_mask=(1, 1, 1),
    )
    first_adapter = tmp_path / "adapter_000"
    second_adapter = tmp_path / "adapter_001"

    train_pappo_ppo_update(
        model_name="tiny-local",
        samples=[sample],
        output_dir=first_adapter,
        local_files_only=True,
    )
    result = train_pappo_ppo_update(
        model_name="tiny-local",
        samples=[sample],
        output_dir=second_adapter,
        input_adapter_path=first_adapter,
        local_files_only=True,
    )

    assert result.status == "pappo_ppo_update_completed"
    assert result.adapter_dir == str(second_adapter)
    assert (second_adapter / "adapter_config.json").exists()


def test_fill_samples_with_model_logprobs_populates_old_ref_and_mask() -> None:
    sample = PPOTurnSample(
        trajectory_id="t1",
        turn_id=0,
        tool_name="edit",
        prompt="<task>",
        action_text="def return",
        target=1.0,
        value=0.0,
        old_logprobs=(),
        ref_logprobs=(),
        action_mask=(),
    )

    filled = fill_samples_with_model_logprobs(
        model_name="tiny-local",
        samples=[sample],
        old_adapter_path=None,
        max_length=64,
        local_files_only=True,
    )

    assert len(filled) == 1
    assert filled[0].old_logprobs
    assert filled[0].ref_logprobs
    assert filled[0].action_mask
    assert len(filled[0].old_logprobs) == len(filled[0].ref_logprobs)
    assert len(filled[0].old_logprobs) == len(filled[0].action_mask)
    assert sum(filled[0].action_mask) > 0


def test_apply_advantage_prior_preserves_counterfactual_and_adds_small_prior() -> None:
    samples = [
        PPOTurnSample(
            trajectory_id="task-a",
            turn_id=0,
            tool_name="edit",
            prompt="<task>",
            action_text="good",
            target=1.0,
            value=0.0,
            old_logprobs=(-0.7,),
            ref_logprobs=(-0.7,),
            action_mask=(1,),
        ),
        PPOTurnSample(
            trajectory_id="task-a",
            turn_id=0,
            tool_name="edit",
            prompt="<task>",
            action_text="bad",
            target=-1.0,
            value=0.0,
            old_logprobs=(-0.7,),
            ref_logprobs=(-0.7,),
            action_mask=(1,),
        ),
        PPOTurnSample(
            trajectory_id="task-b",
            turn_id=0,
            tool_name="edit",
            prompt="<task>",
            action_text="also good",
            target=1.0,
            value=1.0,
            old_logprobs=(-0.7,),
            ref_logprobs=(-0.7,),
            action_mask=(1,),
        ),
    ]

    adjusted = apply_advantage_prior(samples, prior=0.25)

    assert adjusted[0].target - adjusted[0].value == 1.25
    assert adjusted[1].target - adjusted[1].value == -1.25
    assert adjusted[2].target - adjusted[2].value == 0.25
