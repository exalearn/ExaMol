"""Test very simple selection options"""
import numpy as np

from examol.select.random import RandomSelector


def test_random():
    selector = RandomSelector(to_select=2)
    selector.add_possibilities(keys=[1, 2, 3], samples=np.array([1, 2, 3]))
    selector.start_dispensing()
    assert len(list(selector.dispense())) == 2
    assert len(next(selector.dispense())) == 2
