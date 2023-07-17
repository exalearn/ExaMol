"""Tests for the BOTorch-based acquisition functions"""
import numpy as np
from botorch.acquisition import qExpectedImprovement

from examol.select.botorch import BOTorchSequentialSelector


def test_sequential(test_data, recipe):
    """Use EI as a simple test case"""
    x, y, db = test_data

    def update_fn(obs: np.ndarray, options: dict) -> dict:
        options['best_f'] = max(obs)
        return options

    selector = BOTorchSequentialSelector(qExpectedImprovement, acq_options={'best_f': 0.5},
                                         acq_options_updater=update_fn, to_select=1)
    selector.update(db, recipe)
    assert selector.acq_options['best_f'] == max(recipe.lookup(r) for r in db.values())

    # Test maximization
    selector.add_possibilities(x, y)
    sel_x, sel_y = zip(*selector.dispense())
    assert (np.abs(np.subtract(sel_x, 0.5)) < 0.1).all()

    # Test for minimize
    selector.maximize = False
    selector.acq_options_updater = lambda ob, op: update_fn(-ob, op)
    selector.update(db, recipe)
    selector.start_gathering()
    selector.add_possibilities(x, -y)
    sel_x, sel_y = zip(*selector.dispense())
    assert (np.abs(np.subtract(sel_x, 0.5)) < 0.1).all()
