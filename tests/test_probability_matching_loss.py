import pytest
import torch

from sevir_nowcasting.losses import ProbabilityMatchingLoss


def test_identical_fields_have_zero_pm_loss() -> None:
    loss_fn = ProbabilityMatchingLoss()
    field = torch.rand(2, 3, 1, 8, 8)
    assert torch.equal(loss_fn(field, field), torch.tensor(0.0))


def test_pm_loss_is_invariant_to_spatial_permutation() -> None:
    loss_fn = ProbabilityMatchingLoss()
    target = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    prediction = target.flip(-1)
    assert torch.equal(loss_fn(prediction, target), torch.tensor(0.0))


def test_pm_loss_penalizes_distribution_mismatch_and_has_finite_gradient() -> None:
    loss_fn = ProbabilityMatchingLoss()
    prediction = torch.zeros(1, 2, 1, 4, 4, requires_grad=True)
    target = torch.ones_like(prediction)
    loss = loss_fn(prediction, target)
    assert float(loss.detach()) == pytest.approx(1.0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_pm_loss_rejects_non_video_tensor() -> None:
    with pytest.raises(ValueError, match="expected"):
        ProbabilityMatchingLoss()(torch.zeros(2, 3), torch.zeros(2, 3))
