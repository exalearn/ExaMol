"""Tests for the BOTorch-based acquisition functions"""
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.sampling import SobolQMCNormalSampler
from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils import draw_sobol_samples
import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning

from examol.select.botorch import BOTorchSequentialSelector

from conftest import TestRecipe
from examol.store.models import MoleculeRecord


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


def test_evhi():
    # Make two recipes
    test_recipes = [TestRecipe('a', 'test'), TestRecipe('b', 'test')]

    # Set up the problem and the initial training set
    problem = BraninCurrin(negate=True)
    train_x = draw_sobol_samples(bounds=problem.bounds, n=8, q=1).squeeze(1)
    train_y = problem(train_x)
    train_mols = {}
    for i, y in enumerate(train_y):
        record = MoleculeRecord.from_identifier('C' * (i + 1))
        record.properties = {'a': {'test': y[0]}, 'b': {'test': y[1]}}
        train_mols[record.key] = record

    # Make some test points
    sample_x = draw_sobol_samples(bounds=problem.bounds, n=32, q=1).squeeze(1)
    sample_y = problem(sample_x).T  # Will be (num objectives) x (num samples)
    sample_y = torch.unsqueeze(sample_y, dim=-1)
    sample_y = torch.tile(sample_y, dims=[1, 1, 8])
    sample_y += torch.rand(*sample_y.shape) * 0.1

    # Create the sampler
    def update_fn(selector: BOTorchSequentialSelector, obs: np.ndarray) -> dict:
        options = selector.acq_options.copy()
        options['ref_point'] = obs.min(axis=0)
        options['partitioning'] = FastNondominatedPartitioning(
            ref_point=problem.ref_point,
            Y=torch.from_numpy(obs),
        )
        return options

    selector = BOTorchSequentialSelector(qExpectedHypervolumeImprovement,
                                         acq_options={'sampler': SobolQMCNormalSampler(sample_shape=torch.Size([16]))},
                                         acq_options_updater=update_fn,
                                         to_select=1)

    selector.update(train_mols, test_recipes)

    # Test maximization
    selector.add_possibilities(sample_x, sample_y.detach().numpy())
    sel_x, sel_y = zip(*selector.dispense())
