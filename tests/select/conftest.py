from botorch.test_functions.multi_objective import BraninCurrin
from botorch.utils import draw_sobol_samples
from pytest import fixture
import numpy as np
import torch

from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


class TestRecipe(PropertyRecipe):
    pass


@fixture()
def recipe() -> PropertyRecipe:
    return TestRecipe('test', 'test')


@fixture()
def test_data():
    # New points
    x = np.linspace(0, 1, 32)
    y = x * (1 - x)
    y = np.random.normal(scale=0.01, size=(32, 8)) + y[None, :, None]

    # Example database
    record = MoleculeRecord.from_identifier('C')
    record.properties['test'] = {'test': 0.25}
    record_2 = MoleculeRecord.from_identifier('O')
    return x, y, {record.key: record, record_2.key: record_2}


@fixture()
def multi_recipes() -> list[PropertyRecipe]:
    return [TestRecipe('a', 'test'), TestRecipe('b', 'test')]


@fixture()
def multi_test_data() -> tuple[np.ndarray, np.ndarray, dict[str, MoleculeRecord]]:
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

    return sample_x.numpy(), sample_y.numpy(), train_mols
