from pytest import fixture
import numpy as np

from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


@fixture()
def recipe() -> PropertyRecipe:
    class TestRecipe(PropertyRecipe):
        def __init__(self):
            super().__init__('test', 'test')

    return TestRecipe()


@fixture()
def test_data():
    # New points
    x = np.linspace(0, 1, 32)
    y = x * (1 - x)
    y = np.random.normal(scale=0.001, size=(32, 8)) + y[:, None]

    # Example database
    record = MoleculeRecord.from_identifier('C')
    record.properties['test'] = {'test': 0.25}
    return x, y, {record.key: record}
