import numpy as np
from pytest import fixture

from examol.select.bayes import ExpectedImprovement
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


class TestRecipe(PropertyRecipe):
    def __init__(self):
        super().__init__('test', 'test')


@fixture()
def data():
    # New points
    x = np.linspace(0, 1, 32)
    y = x * (1 - x)
    y = np.random.normal(scale=0.001, size=(32, 8)) + y[:, None]

    # Example database
    record = MoleculeRecord.from_identifier('C')
    record.properties['test'] = {'test': 0.25}
    return x, y, {record.key: record}


def test_ei(data):
    x, y, db = data
    sel = ExpectedImprovement(2, maximize=True, epsilon=0.01)
    sel.update(db, TestRecipe())
    assert sel.best_so_far == 0.25

    # Test it for maximize
    sel.start_gathering()
    sel.add_possibilities(x, y)
    sel.start_dispensing()
    sel_x, sel_y = zip(*sel.dispense())
    assert (np.abs(np.subtract(sel_x, 0.5)) < 0.1).all()

    # Test for minimize
    sel.maximize = False
    sel.best_so_far *= -1
    sel.start_gathering()
    sel.add_possibilities(x, -y)
    sel.start_dispensing()
    sel_x, sel_y = zip(*sel.dispense())
    assert (np.abs(np.subtract(sel_x, 0.5)) < 0.1).all()
