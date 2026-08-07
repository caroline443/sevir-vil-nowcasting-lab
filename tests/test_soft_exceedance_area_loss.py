import torch

from sevir_nowcasting.losses import SoftExceedanceAreaLoss


def test_identical_fields_have_zero_loss() -> None:
    loss_fn = SoftExceedanceAreaLoss()
    field = torch.rand(2, 3, 1, 16, 16)
    assert torch.equal(loss_fn(field, field), torch.tensor(0.0))


def test_erasing_a_severe_core_is_penalized() -> None:
    loss_fn = SoftExceedanceAreaLoss()
    target = torch.zeros(1, 2, 1, 16, 16)
    target[:, :, :, 4:8, 5:9] = 230.0 / 255.0
    erased = torch.zeros_like(target)
    assert loss_fn(erased, target) > 0.1


def test_spatial_translation_does_not_change_area_loss() -> None:
    loss_fn = SoftExceedanceAreaLoss()
    target = torch.zeros(1, 1, 1, 16, 16)
    prediction = torch.zeros_like(target)
    target[:, :, :, 2:6, 3:7] = 230.0 / 255.0
    prediction[:, :, :, 9:13, 8:12] = 230.0 / 255.0
    assert torch.allclose(loss_fn(prediction, target), torch.tensor(0.0), atol=1e-7)


def test_gradient_is_finite() -> None:
    loss_fn = SoftExceedanceAreaLoss()
    prediction = torch.rand(1, 2, 1, 8, 8, requires_grad=True)
    target = torch.rand_like(prediction)
    loss_fn(prediction, target).backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_sequence_mean_ignores_leadwise_area_exchange() -> None:
    target = torch.zeros(1, 2, 1, 8, 8)
    prediction = torch.zeros_like(target)
    target[:, 0, :, 1:5, 1:5] = 230.0 / 255.0
    prediction[:, 1, :, 1:5, 1:5] = 230.0 / 255.0
    per_lead = SoftExceedanceAreaLoss(temporal_mode="per_lead")
    sequence_mean = SoftExceedanceAreaLoss(temporal_mode="sequence_mean")
    assert per_lead(prediction, target) > 0
    assert torch.allclose(
        sequence_mean(prediction, target), torch.tensor(0.0), atol=1e-7
    )


def test_unknown_temporal_mode_is_rejected() -> None:
    try:
        SoftExceedanceAreaLoss(temporal_mode="unknown")
    except ValueError as exc:
        assert "temporal_mode" in str(exc)
    else:
        raise AssertionError("unknown temporal mode was accepted")
