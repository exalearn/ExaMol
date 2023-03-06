"""Test for different recipes"""
from math import isclose

from ase import units

from examol.store.recipes import SolvationEnergy


def test_solvation(record, sim_result):
    # Add the vacuum energy
    record.add_energies(sim_result)

    # Add the energy in the solvent
    sim_result.solvent = 'acn'
    sim_result.energy -= units.kcal / units.mol
    record.add_energies(sim_result)

    # Compute the solvation energy
    recipe = SolvationEnergy('test', 'acn')
    assert recipe.level == 'test_acn'
    assert isclose(recipe.compute_property(record), -1)
    recipe.update_record(record)
    assert isclose(record.properties['solvation_energy']['test_acn'], -1)
