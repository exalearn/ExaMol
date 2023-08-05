import os
from math import isclose

from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.build import molecule
from pytest import mark, fixture
import numpy as np

from examol.simulate.ase.utils import make_ephemeral_calculator, initialize_charges, add_vacuum_buffer


@fixture()
def atoms():
    atoms = molecule('H2O')
    atoms.center(vacuum=5)
    return atoms


@mark.parametrize('calc', [
    LennardJones(),
    {'name': 'cp2k', 'kwargs': {'label': 'test'}},
    {'name': 'xtb'}
])
def test_make(calc, atoms, tmpdir):
    os.chdir(tmpdir)
    with make_ephemeral_calculator(calc) as ase_calc:
        assert isinstance(ase_calc, Calculator)
        ase_calc.get_potential_energy(atoms)
    assert ase_calc is not None


def test_charges(atoms):
    initialize_charges(atoms, 1)
    assert isclose(atoms.get_initial_charges().sum(), 1)
    initialize_charges(atoms, 0)
    assert isclose(atoms.get_initial_charges().sum(), 0)


def test_buffer(atoms):
    # The closest atom to each side should be exactly 2 Ang
    add_vacuum_buffer(atoms, buffer_size=2, cubic=False)
    assert np.isclose(atoms.positions.min(axis=0), 2.).all()
    assert np.isclose(atoms.cell.max(axis=0) - atoms.positions.max(axis=0), 2.).all()

    # The closest atom should be at least 2 Ang
    add_vacuum_buffer(atoms, buffer_size=2, cubic=True)
    assert np.isclose(atoms.cell.lengths()[0], atoms.cell.lengths()).all()
    assert np.greater_equal(atoms.positions.min(axis=0), 2.).all()
    assert np.greater_equal(atoms.cell.max(axis=0) - atoms.positions.max(axis=0), 2.).all()
