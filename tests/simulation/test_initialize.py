import numpy as np

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
