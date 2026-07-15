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
