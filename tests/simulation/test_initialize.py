import numpy as np

from examol.simulate.initialize import add_initial_conformer
from examol.store.models import MoleculeRecord, Conformer
from examol.utils.conversions import read_from_string
from examol.simulate import initialize as init


def test_geoms():
    inchi, xyz = init.generate_inchi_and_xyz("C")
    assert xyz.startswith('5')
    assert inchi == 'InChI=1S/CH4/h1H4'

    # Make sure it flattens as a cyclopropenium ion
    inchi, xyz = init.generate_inchi_and_xyz("c1c[c+]1Cl")
    assert xyz.startswith('6')
    assert inchi == 'InChI=1S/C3H2Cl/c4-3-1-2-3/h1-2H/q+1'
    atoms = read_from_string(xyz, 'xyz')
    for i in range(3, 6):  # For each atom attached to the ion
        assert np.linalg.det(atoms.positions[:3, :] - atoms.positions[i, :]) < 1e-6  # Test for coplanar


def test_record():
    # Test adding a conformer
    record = MoleculeRecord.from_identifier('C')
    assert len(record.conformers) == 0

    assert add_initial_conformer(record) is record  # Return self
    assert len(record.conformers) == 1
    orig_conf = record.conformers[0]

    # Add a perturbed geometry, make sure it has a higher energy
    new_conf = Conformer.from_xyz(
        orig_conf.xyz.replace('C      -0.0126', 'C      -0.0200'),
        source='perturb',
        config_name='mmff',
        charge=0
    )
    record.conformers.append(new_conf)
    add_initial_conformer(record)
    assert all(len(x.energies) == 1 for x in [new_conf, orig_conf])
    assert new_conf.energies[0].energy > orig_conf.energies[0].energy

    # Make sure it does not add energies to charged molecules
    charged_conf = Conformer.from_xyz(
        orig_conf.xyz.replace('C      -0.0126', 'C      -0.0300'),
        source='perturb',
        config_name='mmff',
        charge=1
    )
    record.conformers.append(charged_conf)
    add_initial_conformer(record)
    assert all(len(x.energies) == 1 for x in [new_conf, orig_conf])
    assert len(charged_conf.energies) == 0
