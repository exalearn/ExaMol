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
    y = np.random.normal(scale=0.01, size=(32, 8)) + y[None, :, None]

    # Example database
    record = MoleculeRecord.from_identifier('C')
    record.properties['test'] = {'test': 0.25}
    record_2 = MoleculeRecord.from_identifier('O')
    return x, y, {record.key: record, record_2.key: record_2}
