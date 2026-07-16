import torch

from scripts.diagnose_spatial_false_alarms import dilate, safe_ratio


def test_dilate_expands_mask_by_requested_radius() -> None:
    mask = torch.zeros(1, 1, 1, 7, 7, dtype=torch.bool)
    mask[..., 3, 3] = True
    expanded = dilate(mask, radius=2)
    assert int(expanded.sum()) == 25
    assert expanded[..., 1:6, 1:6].all()


def test_zero_radius_preserves_mask() -> None:
    mask = torch.rand(2, 3, 1, 8, 8) > 0.5
    assert torch.equal(dilate(mask, radius=0), mask)


def test_safe_ratio_maps_empty_forecasts_to_zero() -> None:
    result = safe_ratio(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 2.0]))
    assert result == [0.0, 0.5]

