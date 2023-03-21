"""Test for different recipes"""
import copy
from math import isclose

from pytest import raises
from ase import units

from examol.store.recipes import SolvationEnergy, RedoxEnergy


def test_solvation(record, sim_result):
    """Test solvation and the basic properties of a Recipe"""
    recipe = SolvationEnergy('test', 'acn')

    # Make sure it returns `None`, as we do not have this energy yet
    assert recipe.lookup(record) is None
    assert recipe.lookup(record, recompute=True) is None

    # See that the next suggestion is a relaxation
    requests = recipe.suggest_computations(record)
    assert len(requests) == 1
    assert requests[0].optimize
    assert requests[0].solvent is None

    # Add the vacuum energy
    record.add_energies(sim_result)

    # See that the next suggestion is a solvent computation
    requests = recipe.suggest_computations(record)
    assert len(requests) == 1
    assert not requests[0].optimize
    assert requests[0].solvent == 'acn'

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

    # Ensure there is nothing else to do
    requests = recipe.suggest_computations(record)
    assert len(requests) == 0


def test_redox(record, sim_result):
    recipe = RedoxEnergy(1, 'test', vertical=True)
    requests = recipe.suggest_computations(record)
    assert len(requests) == 1
    assert requests[0].optimize
    assert requests[0].charge == 0

    # Add the vacuum energy,
    assert record.add_energies(sim_result)
    requests = recipe.suggest_computations(record)
    assert len(requests) == 1
    assert not requests[0].optimize
    assert requests[0].charge == 1

    # Now add charged energy of the same molecule
    sim_result.charge = 1
    sim_result.energy += 1
    assert not record.add_energies(sim_result)
    requests = recipe.suggest_computations(record)
    assert len(requests) == 0

    # Make sure we can compute a vertical
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

    requests = recipe.suggest_computations(record)
    assert len(requests) == 1
    assert requests[0].optimize
    assert requests[0].charge == 1
    assert requests[0].xyz == sim_result.xyz

    # Add a different geometry
    adia_sim_result = copy.copy(sim_result)
    adia_sim_result.xyz = sim_result.xyz.replace('0.000', '0.010')
    adia_sim_result.energy -= 0.5
    assert record.add_energies(adia_sim_result)
    assert len(record.conformers) == 2
    assert isclose(recipe.compute_property(record), 0.5)

    requests = recipe.suggest_computations(record)
    assert len(requests) == 0

    # Start assessing the energy in solvent
    recipe = RedoxEnergy(1, 'test', solvent='acn', vertical=False)

    requests = recipe.suggest_computations(record)
    assert len(requests) == 2
    assert not any(x.optimize for x in requests)
    assert any(x.solvent == 'acn' for x in requests)
    assert requests[0].charge == 0 and requests[0].xyz == sim_result.xyz
    assert requests[1].charge == 1 and requests[1].xyz == adia_sim_result.xyz

    adia_sim_result.solvent = 'acn'
    adia_sim_result.energy -= 2
    assert not record.add_energies(adia_sim_result)

    sim_result.charge = 0
    sim_result.solvent = 'acn'
    sim_result.energy -= 2
    assert not record.add_energies(sim_result)

    assert 'acn' in recipe.level
    assert isclose(recipe.compute_property(record), -0.5)

    requests = recipe.suggest_computations(record)
    assert len(requests) == 0


def test_adia_redox(record):
    """Make sure we can cold-start an adiabatic redox energy computation"""
    recipe = RedoxEnergy(1, 'test', vertical=False)
    suggestions = recipe.suggest_computations(record)
    assert len(suggestions) == 2
