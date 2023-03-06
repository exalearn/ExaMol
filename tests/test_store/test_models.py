from math import isclose

from pytest import raises

from examol.store.models import Conformer, MoleculeRecord


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


def test_find_lowest_conformer(record, sim_result):
    # Make a record with a single energy evaluation
    record.add_energies(sim_result)

    # Find the energy
    conf, energy = record.find_lowest_conformer('test', 0, None)
    assert isclose(energy, -1)
    assert isclose(conf.get_energy('test', 0, None), energy)

    # Add a second conformer, with a higher energy
    sim_result.energy = 0
    sim_result.xyz = sim_result.xyz.replace("0.000", "0.01")
    assert record.add_energies(sim_result)
    assert len(record.conformers) == 2
    conf, energy = record.find_lowest_conformer('test', 0, None)
    assert isclose(energy, -1)
    assert "0.000" in conf.xyz

    # Make sure we do not re-add the second conformer
    assert not record.add_energies(sim_result)

    # Test without a match
    with raises(ValueError) as error:
        record.find_lowest_conformer('not_done', 0, None)
    assert 'not_done' in str(error)


def test_properties(record):
    assert len(record.properties) == 0
