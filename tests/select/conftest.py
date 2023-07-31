from pytest import fixture
import numpy as np

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
    y = np.random.normal(scale=0.001, size=(32, 8)) + y[None, :, None]

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
    # Make training records have properties of o1 = 1 - o2
    objective_1 = np.linspace(0, 1, 8)
    objective_2 = 1 - objective_1
    train_mols = {}
    for i, (y1, y2) in enumerate(zip(objective_1, objective_2)):
        record = MoleculeRecord.from_identifier('C' * (i + 1))
        record.properties = {'a': {'test': y1}, 'b': {'test': y2}}
        train_mols[record.key] = record

    # Make the test points lie along o1^2 + o2^2 = 0.9 (lies outside only for the minimi
    theta = np.linspace(0, np.pi / 2, 16)
    true_y = np.array([0.8 * np.cos(theta), 0.8 * np.sin(theta)])
    sample_y = np.repeat(true_y[:, :, None], repeats=8, axis=-1)
    sample_y += 0.001 * np.random.random(sample_y.shape)
    return theta, sample_y, train_mols
