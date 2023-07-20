"""Tests for the BOTorch-based acquisition functions"""
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.sampling import SobolQMCNormalSampler
import numpy as np
import torch


from examol.select.botorch import BOTorchSequentialSelector


def test_sequential(test_data, recipe):
    """Use EI as a simple test case"""
    x, y, db = test_data

    def update_fn(selector: BOTorchSequentialSelector, obs: np.ndarray) -> dict:
        options = selector.acq_options.copy()
        options['best_f'] = max(obs) if selector.maximize else min(obs)
        return options

    selector = BOTorchSequentialSelector(ExpectedImprovement,
                                         acq_options={'best_f': 0.5},
                                         acq_options_updater=update_fn,
                                         to_select=1)
    selector.update(db, [recipe])
    assert selector.acq_options['best_f'] == 0.25

    # Test maximization
    selector.add_possibilities(x, y)
    sel_x, sel_y = zip(*selector.dispense())
    assert (np.abs(np.subtract(sel_x, 0.5)) < 0.1).all()

    # Test for minimize
    selector.maximize = False
    selector.update(db, [recipe])
    selector.start_gathering()
    selector.add_possibilities(x, -y)
    sel_x, sel_y = zip(*selector.dispense())
    assert (np.abs(np.subtract(sel_x, 0.5)) < 0.1).all()


def test_evhi(multi_test_data, multi_recipes):
    sample_x, sample_y, train_mols = multi_test_data

    # Create the sampler
    def update_fn(selector: BOTorchSequentialSelector, obs: np.ndarray) -> dict:
        options = selector.acq_options.copy()
        options['ref_point'] = obs.min(axis=0)
        options['partitioning'] = FastNondominatedPartitioning(
            ref_point=torch.from_numpy(options['ref_point']),
            Y=torch.from_numpy(obs),
        )
        return options

    selector = BOTorchSequentialSelector(qExpectedHypervolumeImprovement,
                                         acq_options={'sampler': SobolQMCNormalSampler(sample_shape=torch.Size([16]))},
                                         acq_options_updater=update_fn,
                                         to_select=2)

    selector.update(train_mols, multi_recipes)

    # Test maximization
    selector.add_possibilities(sample_x, sample_y)
    sel_x, sel_y = zip(*selector.dispense())
    assert sel_y[0] > sel_y[1]
