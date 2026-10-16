"""Test the credit arithmetic that the trainer calls."""

from __future__ import annotations

import pytest

from shipyard.training.credit import overlong


def test_the_overlong_term_ramps_over_the_last_fifth_of_the_budget_as_eq_13() -> None:
    lengths = [0, 80, 90, 100, 120]
    found = overlong(lengths, penalty=0.5, buffer=0.2, budget=100)
    assert found == pytest.approx([0.0, 0.0, -0.25, -0.5, -0.5])
    assert overlong(lengths, penalty=1.0, buffer=1.0, budget=100) == pytest.approx(
        [0.0, -0.8, -0.9, -1.0, -1.0]
    )
    assert overlong(lengths, penalty=0.5, buffer=0.2, budget=None) == [0.0] * 5
    assert overlong(lengths, penalty=0.0, buffer=0.2, budget=100) == [0.0] * 5
