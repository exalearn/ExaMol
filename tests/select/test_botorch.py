"""Tests for the BOTorch-based acquisition functions"""
from botorch.acquisition import ExpectedImprovement
import numpy as np
import torch

from examol.select.botorch import BOTorchSequentialSelector, EHVISelector, EnsembleCovarianceModel


def test_ensemble_covar():
    # Normal case: non-zero variance for all properties
    x = torch.tensor([[[1.0, 1.1, 2.0, 2.1]]])
    model = EnsembleCovarianceModel(num_outputs=2)
    covar = model.posterior(x)
    assert torch.isclose(covar.distribution.mean, torch.tensor([1.05, 2.05])).all()
    assert torch.isclose(covar.distribution.variance, torch.tensor([[0.0050, 0.0050]]), atol=1e-4).all()

    # Bad case: zero variance
    x = torch.tensor([[[1.0, 1.0, 2.1, 2.1]]])
    covar = model.posterior(x)
    assert torch.isclose(covar.distribution.mean, torch.tensor([1.0, 2.1])).all()
    assert torch.isclose(covar.distribution.variance, torch.tensor([[0.0, 0.0]]), atol=1e-4).all()


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
    selector = EHVISelector(to_select=2)
    selector.update(train_mols, multi_recipes)
    assert np.isclose(selector.acq_options['ref_point'], [0., 0.]).all()

    # Test maximization
    selector.add_possibilities(sample_x, sample_y)
    sel_x, sel_y = zip(*selector.dispense())
    assert sel_y[0] > sel_y[1]
    assert (np.abs(np.subtract(np.pi / 4, sel_x)) < 0.2).all()  # Best points are near Pi/4

    # Test minimizing both
    selector.maximize = False
    sample_y *= -1  # Update the samples
    for record in train_mols.iterate_over_records():
        record.properties['a']['test'] *= -1
        record.properties['b']['test'] *= -1

    selector.update(train_mols, multi_recipes)
    assert np.isclose(selector.acq_options['ref_point'], [0., 0.]).all()

    selector.add_possibilities(sample_x, sample_y)
    sel_x, sel_y = zip(*selector.dispense())
    assert sel_y[0] > sel_y[1]
    assert (np.abs(np.subtract(np.pi / 4, sel_x)) < 0.2).all()  # Best points are near Pi/4

    # Test maximizing one objective and minimizing the other
    sample_y[0, :, :] *= -1
    selector.maximize = [True, False]
    for record in train_mols.iterate_over_records():
        record.properties['a']['test'] *= -1

    selector.update(train_mols, multi_recipes)
    assert np.isclose(selector.acq_options['ref_point'], [0., 0.]).all()

    selector.add_possibilities(sample_x, sample_y)
    sel_x, sel_y = zip(*selector.dispense())
    assert sel_y[0] > sel_y[1]
    assert (np.abs(np.subtract(np.pi / 4, sel_x)) < 0.2).all()  # Best points are near Pi/4
