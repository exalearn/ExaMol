from unittest.mock import patch

from ase.db import connect
from pytest import mark, fixture
from ase.build import molecule
from ase.calculators.lj import LennardJones

from examol.simulate.ase import ASESimulator
from examol.simulate.ase.utils import write_to_string


class FakeCP2K(LennardJones):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __del__(self):
        return


@fixture()
def strc() -> str:
    atoms = molecule('H2O')
    return write_to_string(atoms, 'xyz')


@mark.parametrize('config_name', ['cp2k_blyp_szv'])
def test_ase(config_name: str, strc, tmpdir):
    with patch('ase.calculators.cp2k.CP2K', new=FakeCP2K):
        db_path = str(tmpdir / 'data.db')
        sim = ASESimulator(scratch_dir=tmpdir, ase_db_path=db_path)
        out_res, traj_res, extra = sim.optimize_structure(strc, config_name, charge=0)
        assert out_res.energy < traj_res[0].energy

        # Make sure everything is stored in the DB
        with connect(db_path) as db:
            assert len(db) == len(traj_res)
