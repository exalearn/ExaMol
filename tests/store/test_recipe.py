"""Test for different recipes"""
import copy
from math import isclose

from pytest import raises
from ase import units

from examol.store.recipes import SolvationEnergy, RedoxEnergy


def test_solvation(record, sim_result):
    """Test solvation and the basic properties of a Recipe"""
    recipe = SolvationEnergy('test', 'acn')
    assert recipe.lookup(record) is None
    assert recipe.lookup(record, recompute=True) is None

    # Add the vacuum energy
    record.add_energies(sim_result)

    # Add the energy in the solvent
    sim_result.solvent = 'acn'
    sim_result.energy -= units.kcal / units.mol
    record.add_energies(sim_result)

    # Compute the solvation energy
    assert recipe.level == 'test_acn'
    assert isclose(recipe.compute_property(record), -1)
    recipe.update_record(record)
    assert isclose(record.properties['solvation_energy']['test_acn'], -1)
    assert recipe.lookup(record) is not None
    assert recipe.lookup(record, recompute=True) is not None


def test_redox(record, sim_result):
    # Add the vacuum energy, energy in solvent and a charged energy of the same molecule
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
    adia_sim_result = copy.copy(sim_result)
    adia_sim_result.xyz = sim_result.xyz.replace('0.000', '0.010')
    adia_sim_result.energy -= 0.5
    assert record.add_energies(adia_sim_result)
    assert len(record.conformers) == 2
    assert isclose(recipe.compute_property(record), 0.5)

    # Add energies in solvents
    adia_sim_result.solvent = 'acn'
    adia_sim_result.energy -= 2
    assert not record.add_energies(adia_sim_result)

    sim_result.charge = 0
    sim_result.solvent = 'acn'
    sim_result.energy -= 2
    assert not record.add_energies(sim_result)

    recipe = RedoxEnergy(1, 'test', solvent='acn', vertical=False)
    assert 'acn' in recipe.level
    assert isclose(recipe.compute_property(record), -0.5)
