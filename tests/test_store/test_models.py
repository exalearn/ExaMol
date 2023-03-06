from math import isclose

from pytest import fixture, raises
from ase.build import molecule

from examol.utils.conversions import write_to_string
from examol.simulate.base import SimResult
from examol.store.models import Conformer, MoleculeRecord


@fixture()
def sim_result() -> SimResult:
    mol = molecule('CH4')
    return SimResult(
        xyz=write_to_string(mol, 'xyz'),
        charge=0,
        energy=-1,
        config_name='test',
        solvent=None
    )


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


def test_add_conformer(sim_result):
    # Make a record for methane
    record = MoleculeRecord.from_identifier('C')

    # Test adding an energy record
    assert record.add_energies(sim_result)
    assert not record.add_energies(sim_result)
    assert not record.add_energies(sim_result, [sim_result])
    assert len(record.conformers) == 1


def test_find_lowest_conformer(sim_result):
    # Make a record with a single energy evaluation
    record = MoleculeRecord.from_identifier('C')
    record.add_energies(sim_result)

    # Find the energy
    _, energy = record.find_lowest_conformer('test', 0, None)
    assert isclose(energy, -1)

    # Add a second conformer, with a higher energy
    sim_result.energy = 0
    sim_result.xyz = sim_result.xyz.replace("0.000", "0.01")
    assert record.add_energies(sim_result)
    assert len(record.conformers) == 2
    conf, energy = record.find_lowest_conformer('test', 0, None)
    assert isclose(energy, -1)
    assert "0.000" in conf.xyz

    # Test without a match
    with raises(ValueError) as error:
        record.find_lowest_conformer('not_done', 0, None)
    assert 'not_done' in str(error)
