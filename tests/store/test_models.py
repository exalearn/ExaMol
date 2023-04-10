from math import isclose
from copy import copy

from pytest import raises

from examol.simulate.base import SimResult
from examol.store.models import Conformer, MoleculeRecord
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

    # Change the solvent and translate the coordinates
    new_atoms = sim_result.atoms
    new_atoms.translate([1, 1, 1])
    sim_result.xyz = write_to_string(new_atoms, 'xyz')
    sim_result.solvent = 'acn'
    assert conf.add_energy(sim_result)
    assert len(conf.energies) == 4


def test_identifiers():
    # Test SMILES and InChI
    record_1 = MoleculeRecord.from_identifier('C')
    record_2 = MoleculeRecord.from_identifier(inchi=record_1.identifier['inchi'])
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
