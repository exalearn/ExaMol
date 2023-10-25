from math import isclose
from copy import copy

from pytest import raises

from examol.store.models import Conformer, MoleculeRecord
from examol.utils.chemistry import parse_from_molecule_string
from examol.utils.conversions import write_to_string


def test_make_conformer(sim_result):
    # Test making one
    conf = Conformer.from_simulation_result(sim_result, 'relaxation')
    assert len(conf.energies) == 1
    assert conf.energies[0].energy == -1

    # Make sure adding a new conformer doesn't do anything
    assert not conf.add_energy(sim_result)

    # Change the configuration name, make sure it add something else
    sim_result.config_name = 'other_method'
    assert conf.add_energy(sim_result)
    assert len(conf.energies) == 2

    # Change the charge
    sim_result.charge = 1
    assert conf.add_energy(sim_result)
    assert len(conf.energies) == 3

    # Change the solvent
    sim_result.solvent = 'acn'
    assert conf.add_energy(sim_result)
    assert len(conf.energies) == 4


def test_identifiers():
    # Test SMILES and InChI
    record_1 = MoleculeRecord.from_identifier('C')
    record_2 = MoleculeRecord.from_identifier(record_1.identifier.inchi)
    assert record_1.key == record_2.key

    # Make sure a parse can fail
    with raises(ValueError) as error:
        MoleculeRecord.from_identifier('not_real')
    assert 'not_real' in str(error)


def test_add_conformer(record, sim_result):
    # Test adding an energy record
    assert record.add_energies(sim_result)
    assert not record.add_energies(sim_result)
    assert not record.add_energies(sim_result, [sim_result])
    assert len(record.conformers) == 1

    # Test adding optimization in a different charge state
    charged_vert = copy(sim_result)
    charged_vert.charge = 1

    charged_opt = copy(charged_vert)
    charged_opt.xyz = charged_opt.xyz.replace("0.000", "0.015")
    charged_opt.energy -= 1

    assert record.add_energies(charged_opt, [charged_vert, charged_opt])
    assert len(record.conformers) == 2
    assert record.conformers[0].energies[1].charge == 1
    assert record.conformers[0].energies[1].energy == charged_vert.energy


def test_translated_sim(record, sim_result):
    """Ensure that we can still match conformers even if coordinates are translated"""
    assert record.add_energies(sim_result)

    # Move the xyz
    atoms = sim_result.atoms
    atoms.translate([1, 1, 1])
    sim_result.xyz = write_to_string(atoms, 'xyz')
    sim_result.config_name = 'test-2'

    # Make sure we don't add an atom in
    assert not record.add_energies(sim_result)
    assert len(record.conformers[0].energies) == 2


def test_find_lowest_conformer(record, sim_result):
    # Make a record with a single energy evaluation
    record.add_energies(sim_result)

    # Find the energy
    conf, energy = record.find_lowest_conformer('test', 0, None)
    assert isclose(energy, -1)
    assert isclose(conf.get_energy('test', 0, None), energy)

    # Add a second conformer, with a higher energy
    sim_result.energy = 0
    sim_result.xyz = sim_result.xyz.replace("0.000", "0.010")
    assert record.add_energies(sim_result)
    assert len(record.conformers) == 2
    conf, energy = record.find_lowest_conformer('test', 0, None)
    assert isclose(energy, -1)
    assert "0.000" in conf.xyz

    # Make sure we do not re-add the second conformer
    assert not record.add_energies(sim_result)

    # Add a third
    sim_result.xyz = sim_result.xyz.replace("0.010", "0.015")
    assert record.add_energies(sim_result)

    # Test without a match
    with raises(ValueError) as error:
        record.find_lowest_conformer('not_done', 0, None)
    assert 'not_done' in str(error)


def test_properties(record):
    assert len(record.properties) == 0


def test_problematic_smiles():
    record = MoleculeRecord.from_identifier("C1=CC=[N+](C=C1)CCOS(=O)(=O)[O-]")
    assert parse_from_molecule_string(record.identifier.inchi) is not None


def test_match_opt_only(record, sim_result):
    """Test matching conformers with and without filtering by whether they were optimized"""

    # Add energy records for a neutral w/ and w/o charge
    original_energy = sim_result.energy
    assert record.add_energies(sim_result)
    sim_result.charge = 1
    assert not record.add_energies(sim_result)

    # Add an adiabatic geometry, and make it higher in energy
    sim_result.xyz = sim_result.xyz.replace("0.000", "0.010")
    sim_result.energy += 1
    assert record.add_energies(sim_result)

    assert len(record.conformers) == 2

    # Getting the conformer w/o filtering should be the original energy, filtering should yield the new higher energy
    assert record.find_lowest_conformer(sim_result.config_name, 1, None, optimized_only=False)[1] == original_energy
    assert record.find_lowest_conformer(sim_result.config_name, 1, None, optimized_only=True)[1] == sim_result.energy


def test_best_xyz(record, sim_result):
    # Test generating an XYZ
    record.conformers.clear()  # Start with nothing
    conf, new_xyz = record.find_closest_xyz('test', 0)
    assert conf is None

    # Add some conformers
    record.conformers = [
        Conformer.from_xyz(new_xyz, config_name='test', charge=0),
        Conformer.from_xyz(new_xyz, config_name='test', charge=1),
        Conformer.from_xyz(new_xyz, config_name='not_test', charge=-1),
        Conformer.from_xyz(new_xyz, config_name='not_test', charge=0)
    ]

    # We should match to the config_name over charge, and get the closest charge
    matched, _ = record.find_closest_xyz(config_name='test', charge=-1)
    assert matched.config_name == 'test'
    assert matched.charge == 0

    matched, _ = record.find_closest_xyz(config_name='test', charge=2)
    assert matched.config_name == 'test'
    assert matched.charge == 1

    # We should match to the newest if several have the same difference in charge and none match config_name
    matched, _ = record.find_closest_xyz(config_name='another', charge=0)
    assert matched.config_name == 'not_test'
    assert matched.charge == 0
