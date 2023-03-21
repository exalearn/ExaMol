import shutil
from pathlib import Path
from unittest.mock import patch

from ase import units
from ase.db import connect
from pytest import mark, fixture, raises
from ase.build import molecule
from ase.calculators.lj import LennardJones

from examol.simulate.ase import ASESimulator
from examol.utils.conversions import write_to_string


class FakeCP2K(LennardJones):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __del__(self):
        return


@fixture()
def strc() -> str:
    atoms = molecule('H2O')
    return write_to_string(atoms, 'xyz')


def test_config_maker(tmpdir):
    sim = ASESimulator(scratch_dir=tmpdir)

    # Easy example
    config = sim.create_configuration('cp2k_blyp_szv', charge=0, solvent=None)
    assert config['kwargs']['cutoff'] == 600 * units.Ry

    # With a charge
    config = sim.create_configuration('cp2k_blyp_szv', charge=1, solvent=None)
    assert config['kwargs']['cutoff'] == 600 * units.Ry
    assert config['kwargs']['charge'] == 1
    assert config['kwargs']['uks']

    # With an undefined basis set
    with raises(AssertionError):
        sim.create_configuration('cp2k_blyp_notreal', charge=1, solvent=None)

    # For xTB
    config = sim.create_configuration('xtb', charge=0, solvent=None)
    assert config['kwargs'] == {}

    config = sim.create_configuration('xtb', charge=0, solvent='acn')
    assert config['kwargs'] == {'solvent': 'acetonitrile'}


@mark.parametrize('config_name', ['cp2k_blyp_szv'])
def test_optimization(config_name: str, strc, tmpdir):
    with patch('ase.calculators.cp2k.CP2K', new=FakeCP2K):
        db_path = Path(tmpdir) / 'data.db'
        db_path.unlink(missing_ok=True)
        sim = ASESimulator(scratch_dir=tmpdir, ase_db_path=str(db_path), clean_after_run=False)
        out_res, traj_res, extra = sim.optimize_structure(strc, config_name, charge=1)
        assert out_res.energy < traj_res[0].energy

        # Find the output directory
        run_dir = next(Path(tmpdir).glob('ase_opt_*'))
        assert run_dir.is_dir()

        # Make sure everything is stored in the DB
        with connect(db_path) as db:
            assert len(db) == len(traj_res)
            assert next(db.select())['total_charge'] == 1

        # Make sure it doesn't write new stuff
        sim.optimize_structure(strc, config_name, charge=1)
        with connect(db_path) as db:
            assert len(db) == len(traj_res)
            assert next(db.select())['total_charge'] == 1

        # Make sure it can deal with a bad restart file
        (run_dir / 'lbfgs.traj').write_text('bad')  # Kill the restart file
        sim.optimize_structure(strc, config_name, charge=1)
        with connect(db_path) as db:
            assert len(db) == len(traj_res)
            assert next(db.select())['total_charge'] == 1

        # Make sure it cleans up after itself
        sim.clean_after_run = True
        shutil.rmtree(run_dir)
        sim.optimize_structure(strc, config_name, charge=1)
        with connect(db_path) as db:
            assert len(db) == len(traj_res)
            assert next(db.select())['total_charge'] == 1
        assert not run_dir.is_dir()


def test_solvent(strc, tmpdir):
    """Test running computations with a solvent"""

    with patch('ase.calculators.cp2k.CP2K', new=FakeCP2K):
        # Run a test with a patched executor
        db_path = str(tmpdir / 'data.db')
        sim = ASESimulator(scratch_dir=tmpdir, ase_db_path=db_path, clean_after_run=True)
        config = sim.create_configuration('cp2k_blyp_szv', solvent='acn', charge=0)
        assert 'ALPHA' in config['kwargs']['inp']

        # Make sure there are no directories left
        assert len(list(Path(tmpdir).glob('ase_*'))) == 0

        # Run the calculation
        result, metadata = sim.compute_energy(strc, 'cp2k_blyp_szv', charge=0, solvent='acn')
        assert result.energy

        # Check that the data was added
        with connect(db_path) as db:
            assert db.count() == 1


def test_xtb(tmpdir, strc):
    sim = ASESimulator(scratch_dir=tmpdir)

    # Ensure we get a different single point energy
    neutral, _ = sim.compute_energy(strc, config_name='xtb', charge=0)
    charged, _ = sim.compute_energy(strc, config_name='xtb', charge=1)
    solvated, _ = sim.compute_energy(strc, config_name='xtb', charge=0, solvent='acn')
    assert neutral.energy != charged.energy
    assert neutral.energy != solvated.energy

    # Ensure it relaxes under charge
    charged_opt, _, _ = sim.optimize_structure(strc, config_name='xtb', charge=-1)
    charged_opt_neutral, _ = sim.compute_energy(charged_opt.xyz, config_name='xtb', charge=0)
    assert charged_opt.energy != charged_opt_neutral.energy
