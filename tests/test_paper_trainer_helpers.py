from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_paper_simvp", ROOT / "scripts" / "train_paper_simvp.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("requested", "available", "expected"),
    [(0, 10, 10), (3, 10, 3), (20, 10, 10)],
)
def test_loader_batch_limit(
    requested: int, available: int, expected: int
) -> None:
    assert MODULE.loader_batch_limit(requested, available) == expected


def test_loader_batch_limit_rejects_empty_loader() -> None:
    with pytest.raises(ValueError):
        MODULE.loader_batch_limit(0, 0)


def test_selection_direction() -> None:
    assert MODULE.is_better("mcsi_global", 0.3, 0.2)
    assert not MODULE.is_better("mcsi_global", 0.1, 0.2)
    assert MODULE.is_better("mse", 0.1, 0.2)
    assert not MODULE.is_better("mse", 0.3, 0.2)
    assert MODULE.is_better("mse", 1.0, None)


def test_facl_output_activation_is_sigmoid() -> None:
    values = torch.tensor([-1.0, 0.0, 1.0])
    actual = MODULE.apply_output_activation(values, "facl")
    assert torch.allclose(actual, torch.sigmoid(values))


def test_mse_output_activation_is_identity() -> None:
    values = torch.tensor([-1.0, 0.0, 1.0])
    assert MODULE.apply_output_activation(values, "mse") is values


def test_output_activation_rejects_unknown_loss() -> None:
    with pytest.raises(ValueError, match="unsupported training loss"):
        MODULE.apply_output_activation(torch.zeros(1), "unknown")
