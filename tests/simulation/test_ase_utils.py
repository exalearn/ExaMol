import os
from math import isclose

from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.build import molecule
from pytest import mark, fixture

from examol.simulate.ase.utils import make_ephemeral_calculator, initialize_charges


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
