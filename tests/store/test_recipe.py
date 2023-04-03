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


def test_redox_vacuum(record, sim_result):
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


def test_redox_solvent(record, sim_result):
    """Assess whether we compute the properties of redox in solvent correcly"""
    recipe = RedoxEnergy(1, 'test', solvent='acn', vertical=True)

    # For vertical, we only need a relaxation to start
    requests = recipe.suggest_computations(record)
    assert len(requests) == 1
    assert requests[0].optimize
    assert requests[0].config_name == 'test'
    assert requests[0].charge == 0

    # Once we add that, we will need 2 computations (neutral solvent, charged solvent)
    record.add_energies(sim_result)

    requests = recipe.suggest_computations(record)
    assert len(requests) == 2, requests
    assert not any(r.optimize for r in requests)
    assert all(r.config_name == 'test' for r in requests)
    assert set(r.charge for r in requests) == {0, 1}


def test_adia_redox(record):
    """Make sure we can cold-start an adiabatic redox energy computation"""
    recipe = RedoxEnergy(1, 'test', vertical=False)
    suggestions = recipe.suggest_computations(record)
    assert len(suggestions) == 2
