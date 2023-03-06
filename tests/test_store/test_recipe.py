"""Test for different recipes"""
from math import isclose

from pytest import raises
from ase import units

from examol.store.recipes import SolvationEnergy, RedoxEnergy


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


def test_redox(record, sim_result):
    # Add the vacuum energy and a charged energy of the same molecule
    assert record.add_energies(sim_result)
    sim_result.charge = 1
    sim_result.energy += 1
    assert not record.add_energies(sim_result)

    # Make sure we can compute a vertical
    recipe = RedoxEnergy(1, 'test', vertical=True)
    assert 'vertical' in recipe.level
    assert 'oxid' in recipe.name
    assert isclose(recipe.compute_property(record), 1)

    # Make sure it throws an error for adiabatic
    recipe = RedoxEnergy(1, 'test', vertical=False)
    assert 'adiabatic' in recipe.level
    assert 'oxid' in recipe.name
    with raises(AssertionError) as error:
        recipe.compute_property(record)
    assert 'We do not have a relaxed charged molecule' in str(error)

    # Add a different geometry
    sim_result.xyz = sim_result.xyz.replace('0.000', '0.010')
    sim_result.energy -= 0.5
    assert record.add_energies(sim_result)
    assert len(record.conformers) == 2
    assert isclose(recipe.compute_property(record), 0.5)
