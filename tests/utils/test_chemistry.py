"""Tests for the "chemistry" utilities"""

from examol.utils.chemistry import get_baseline_charge


def test_charge():
    assert get_baseline_charge('O') == 0
    assert get_baseline_charge('[NH4+]') == 1
    assert get_baseline_charge('Fc1c(F)c1=[F+]') == 1
