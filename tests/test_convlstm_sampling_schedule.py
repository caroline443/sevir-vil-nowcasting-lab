from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "train_openstl_convlstm",
    ROOT / "scripts" / "train_openstl_convlstm.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def probability(
    schedule: str,
    completed_updates: int,
    *,
    total_steps: int = 4,
    current_probability: float = 1.0,
) -> float:
    return MODULE.next_teacher_forcing_probability(
        schedule=schedule,
        completed_updates=completed_updates,
        total_steps=total_steps,
        current_probability=current_probability,
        changing_rate=0.1,
        stop_iter=3,
        end_probability=0.0,
    )


def test_budget_linear_reaches_zero_on_final_update() -> None:
    values = [
        probability("budget_linear", update)
        for update in range(4)
    ]
    assert values == pytest.approx([0.75, 0.5, 0.25, 0.0])


def test_upstream_preserves_fixed_decrement_and_stop() -> None:
    first = probability("upstream", 0, current_probability=1.0)
    second = probability("upstream", 1, current_probability=first)
    stopped = probability("upstream", 3, current_probability=second)
    assert first == pytest.approx(0.9)
    assert second == pytest.approx(0.8)
    assert stopped == 0.0


def test_unknown_schedule_is_rejected() -> None:
    with pytest.raises(ValueError):
        probability("unknown", 0)
