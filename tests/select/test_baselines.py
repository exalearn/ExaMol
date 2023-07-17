"""Test very simple selection options"""
import numpy as np

from examol.select.baseline import RandomSelector, GreedySelector


def test_random():
    selector = RandomSelector(to_select=2)
    selector.add_possibilities(keys=[1, 2, 3], samples=np.array([[1, 2, 3]]).T)
    assert len(list(selector.dispense())) == 2
    assert len(next(selector.dispense())) == 2

    # Reset and make sure options are cleared
    selector.start_gathering()
    assert len(list(selector.dispense())) == 0


def test_greedy():
    selector = GreedySelector(to_select=2, maximize=True)

    # Test it maximizing
    selector.add_possibilities(keys=[1, 2, 3], samples=np.array([[1, 2, 3]]).T)
    assert list(selector.dispense()) == [(3, 3.), (2, 2.)]

    # Test it minimizing
    selector.maximize = False
    selector.start_gathering()
    selector.add_possibilities(keys=[1, 2, 3], samples=np.array([[1, 2, 3]]).T)
    assert list(selector.dispense()) == [(1, -1.), (2, -2.)]  # The score is the negative mean
