import random

import pytest
import torch

from sevir_nowcasting.losses import FourierAmplitudeCorrelationLoss


def test_facl_identical_nonzero_fields_have_zero_terms() -> None:
    field = torch.rand(2, 3, 1, 8, 8)
    prediction_fft, target_fft = FourierAmplitudeCorrelationLoss._transforms(
        field, field
    )
    assert float(
        FourierAmplitudeCorrelationLoss._fal(prediction_fft, target_fft)
    ) == pytest.approx(0.0, abs=1e-7)
    assert float(
        FourierAmplitudeCorrelationLoss._fcl(prediction_fft, target_fft)
    ) == pytest.approx(0.0, abs=1e-6)


def test_facl_schedule_reaches_amplitude_only_tail() -> None:
    loss_fn = FourierAmplitudeCorrelationLoss(total_steps=10, constant_ratio=0.2)
    assert loss_fn.fal_probability() == pytest.approx(0.0)
    loss_fn.step = 7
    assert loss_fn.fal_probability() == pytest.approx(1.0)
    loss_fn.step = 8
    assert loss_fn.fal_probability() == pytest.approx(1.0)


def test_facl_gradient_is_finite() -> None:
    random.seed(0)
    loss_fn = FourierAmplitudeCorrelationLoss(total_steps=4)
    raw_prediction = torch.randn(1, 2, 1, 8, 8, requires_grad=True)
    prediction = torch.sigmoid(raw_prediction)
    target = torch.rand_like(prediction)
    loss = loss_fn(prediction, target)
    loss.backward()
    assert raw_prediction.grad is not None
    assert torch.isfinite(raw_prediction.grad).all()
    assert torch.isfinite(loss)
