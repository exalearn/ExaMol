"""Test very simple selection options"""
from pytest import fixture, raises
import numpy as np

from examol.select.baseline import RandomSelector, GreedySelector


@fixture()
def samples() -> np.ndarray:
    return np.arange(3)[None, :, None] + 1


keys = [1, 2, 3]


def test_random(samples):
    selector = RandomSelector(to_select=2)
    selector.add_possibilities(keys=keys, samples=samples)
    assert len(list(selector.dispense())) == 2
    assert len(next(selector.dispense())) == 2

    # Reset and make sure options are cleared
    selector.start_gathering()
    assert len(list(selector.dispense())) == 0


def test_greedy(samples):
    selector = GreedySelector(to_select=2, maximize=True)

    # Test it maximizing
    selector.add_possibilities(keys=keys, samples=samples)
    assert list(selector.dispense()) == [(3, 3.), (2, 2.)]

    # Test it minimizing
    selector.maximize = False
    selector.start_gathering()
    selector.add_possibilities(keys=keys, samples=samples)
    assert list(selector.dispense()) == [(1, -1.), (2, -2.)]  # The score is the negative mean

    # Make sure it throws an error for multiobjective
    samples = np.repeat(samples, repeats=2, axis=0)
    with raises(ValueError) as e:
        selector.add_possibilities(keys=keys, samples=samples)
    assert "multi-objective" in str(e)
