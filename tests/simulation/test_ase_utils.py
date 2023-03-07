import os

from ase.calculators.calculator import Calculator
from ase.calculators.lj import LennardJones
from ase.build import molecule
from pytest import mark, fixture

from examol.simulate.ase.utils import make_ephemeral_calculator, buffer_cell


@fixture()
def atoms():
    atoms = molecule('H2O')
    buffer_cell(atoms)
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
