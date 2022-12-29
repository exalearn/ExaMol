from pytest import mark, fixture
from ase.build import molecule

from examol.simulate.ase import ASESimulator
from examol.simulate.ase.utils import write_to_string


@fixture()
def strc() -> str:
    atoms = molecule('H2O')
    return write_to_string(atoms, 'xyz')


@mark.parametrize('config_name', ['cp2k_blyp_szv'])
def test_ase(config_name: str, strc, tmpdir):
    sim = ASESimulator(scratch_dir=tmpdir)
    out_res, traj_res, extra = sim.optimize_structure(strc, config_name, charge=0)
    assert out_res.energy < traj_res[0].energy
