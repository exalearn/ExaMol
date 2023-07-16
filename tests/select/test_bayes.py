import numpy as np

from examol.select.bayes import ExpectedImprovement


def test_ei(test_data, recipe):
    x, y, db = test_data
    sel = ExpectedImprovement(2, maximize=True, epsilon=0.01)
    sel.update(db, recipe)
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
