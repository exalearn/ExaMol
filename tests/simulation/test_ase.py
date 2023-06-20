import shutil
from pathlib import Path
from unittest.mock import patch

from ase import units
from ase.calculators.gaussian import Gaussian
from ase.db import connect
from pytest import mark, fixture, raises
from ase.build import molecule
from ase.calculators.lj import LennardJones

from examol.simulate.ase import ASESimulator
from examol.simulate.ase.utils import make_ephemeral_calculator
from examol.simulate.initialize import generate_inchi_and_xyz
from examol.utils.conversions import write_to_string


_files_dir = Path(__file__).parent / 'files'


class FakeCP2K(LennardJones):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __del__(self):
        return


@fixture()
def strc() -> str:
    atoms = molecule('H2O')
    return write_to_string(atoms, 'xyz')


def test_cp2k_configs(tmpdir, strc):
    sim = ASESimulator(scratch_dir=tmpdir)

    # Easy example
    config = sim.create_configuration('cp2k_blyp_szv', strc, charge=0, solvent=None)
    assert config['kwargs']['cutoff'] == 600 * units.Ry

    # With a charge
    config = sim.create_configuration('cp2k_blyp_szv', strc, charge=1, solvent=None)
    assert config['kwargs']['cutoff'] == 600 * units.Ry
    assert config['kwargs']['charge'] == 1
    assert config['kwargs']['uks']

    # With an undefined basis set
    with raises(AssertionError):
        sim.create_configuration('cp2k_blyp_notreal', strc, charge=1, solvent=None)


def test_xtb_configs(tmpdir, strc):
    sim = ASESimulator(scratch_dir=tmpdir)
    # For xTB
    config = sim.create_configuration('xtb', strc, charge=0, solvent=None)
    assert config['kwargs'] == {}

    config = sim.create_configuration('xtb', strc, charge=0, solvent='acn')
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
        config = sim.create_configuration('cp2k_blyp_szv', strc, solvent='acn', charge=0)
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


def test_gaussian_configs(strc):
    sim = ASESimulator(gaussian_command='g09')
    assert sim.gaussian_command == 'g09 < PREFIX.com > PREFIX.log'

    # Make a regular configuration
    config = sim.create_configuration('gaussian_b3lyp_6-31g(2df,p)', strc, 0, None)
    assert config['name'] == 'gaussian'
    assert config['kwargs']['method'] == 'b3lyp'
    assert config['kwargs']['basis'] == '6-31g(2df,p)'
    assert config['use_gaussian_opt']

    # Make one with a solvent
    config = sim.create_configuration('gaussian_b3lyp_6-31g(2df,p)', strc, 0, 'acn')
    assert config['kwargs']['SCRF'] == 'PCM,Solvent=acetonitrile'

    # Make one with a charge
    config = sim.create_configuration('gaussian_b3lyp_6-31g(2df,p)', strc, -1, 'acn')
    assert config['kwargs']['charge'] == -1
    assert config['kwargs']['mult'] == 2

    # Make sure extra arguments get passed through
    config = sim.create_configuration('gaussian_b3lyp_6-31g(2df,p)', strc, 0, 'acn', test='yeah')
    assert config['kwargs']['test'] == 'yeah'

    # Make sure it errors as necessary
    with raises(ValueError):
        sim.create_configuration('gaussian_b3lyp_fake_3-21g', strc, 0, None)

    # Make sure the configuration can be mapped to a Gaussian calculator
    with make_ephemeral_calculator(config) as calc:
        assert isinstance(calc, Gaussian)

    # Make a huge structure
    _, big_strc = generate_inchi_and_xyz('C' * 25)
    config = sim.create_configuration('gaussian_b3lyp_6-31g(2df,p)', big_strc, 0, 'acn')
    assert config['kwargs']['ioplist'] == ["2/9=2000"]
    assert not config['use_gaussian_opt']


def test_gaussian_opt(strc, tmpdir):
    sim = ASESimulator(gaussian_command='g16', scratch_dir=tmpdir)

    if shutil.which('g16') is None:
        # Fake execution by having it copy a known output to a target directory
        sim.gaussian_command = f'cp {(_files_dir / "Gaussian-relax.log").absolute()} Gaussian.log'

    relaxed, traj, _ = sim.optimize_structure(strc, 'gaussian_b3lyp_6-31g(2df,p)', charge=0)
    assert relaxed.energy < traj[0].energy
    assert relaxed.forces.max() < 0.01
