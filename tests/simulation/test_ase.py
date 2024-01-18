import itertools
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import units
from ase.calculators.gaussian import Gaussian
from ase.db import connect
from pytest import mark, fixture, raises, param
from ase.build import molecule
from ase.calculators.lj import LennardJones

from examol.simulate.ase import ASESimulator
from examol.simulate.ase.utils import make_ephemeral_calculator
from examol.simulate.initialize import generate_inchi_and_xyz
from examol.utils.conversions import write_to_string

try:
    import xtb  # noqa: F401

    has_xtb = True
except ImportError:
    has_xtb = False

_files_dir = Path(__file__).parent / 'files'

has_cpk2 = shutil.which('cp2k_shell') is not None
is_ci = os.environ.get('CI', None) == "true"

cp2k_configs_to_test = ['cp2k_b3lyp_svp', 'cp2k_blyp_szv', 'cp2k_wb97x_d3_tzvpd']


class FakeShell:

    def __del__(self):
        return


class FakeCP2K(LennardJones):
    _shell = FakeShell()

    def __init__(self, *args, **kwargs):
        super().__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return


@fixture()
def strc() -> str:
    atoms = molecule('H2')
    return write_to_string(atoms, 'xyz')


def test_cp2k_configs(tmpdir, strc):
    sim = ASESimulator(scratch_dir=tmpdir)

    # Easy example
    config = sim.create_configuration('cp2k_blyp_szv', strc, charge=0, solvent=None)
    assert config['kwargs']['cutoff'] == 700 * units.Ry

    # With a charge
    config = sim.create_configuration('cp2k_blyp_szv', strc, charge=1, solvent=None)
    assert config['kwargs']['cutoff'] == 700 * units.Ry
    assert config['kwargs']['charge'] == 1
    assert config['kwargs']['uks']

    # With B3LYP
    config = sim.create_configuration('cp2k_b3lyp_tzvpd', strc, charge=1, solvent=None)
    assert config['kwargs']['cutoff'] == 500 * units.Ry
    assert config['kwargs']['charge'] == 1
    assert config['kwargs']['uks']
    assert 'GAPW' in config['kwargs']['inp']
    assert Path(config['kwargs']['basis_set_file']).is_file()

    # With wb97x-d3
    config = sim.create_configuration('cp2k_wb97x_d3_tzvpd', strc, charge=1, solvent=None)
    assert config['kwargs']['cutoff'] == 600 * units.Ry
    assert config['kwargs']['charge'] == 1
    assert config['kwargs']['uks']
    assert 'GAPW' in config['kwargs']['inp']
    assert Path(config['kwargs']['basis_set_file']).is_file()

    # With an undefined basis set
    with raises(AssertionError):
        sim.create_configuration('cp2k_blyp_notreal', strc, charge=1, solvent=None)


@mark.skipif(is_ci, reason='Too slow for CI')
@mark.skipif(not has_cpk2, reason='CP2K is not installed')
@mark.slow
@mark.parametrize(
    'config,charge,solvent',
    [(xc, c, None) for xc, c in itertools.product(cp2k_configs_to_test, [0, 1])]  # Closed and open shell
    + [(xc, 0, 'acn') for xc in cp2k_configs_to_test]  # With a solvent
    + [(cp2k_configs_to_test[0], -1, 'acn')]  # Open shell and a solvent
)
def test_ase_singlepoint(tmpdir, strc, config, charge, solvent):
    sim = ASESimulator(scratch_dir=tmpdir)
    sim.compute_energy('test', strc, config_name=config, charge=charge, solvent=solvent)


def test_xtb_configs(tmpdir, strc):
    sim = ASESimulator(scratch_dir=tmpdir)
    # For xTB
    config = sim.create_configuration('xtb', strc, charge=0, solvent=None)
    assert config['kwargs'] == {'accuracy': 0.05}

    config = sim.create_configuration('xtb', strc, charge=0, solvent='acn')
    assert config['kwargs'] == {'solvent': 'acetonitrile', 'accuracy': 0.05}


@mark.parametrize('config_name', ['cp2k_blyp_szv', param('xtb', marks=mark.skipif(not has_xtb, reason='xTB is not installed'))])
def test_optimization(config_name: str, strc, tmpdir):
    with patch('ase.calculators.cp2k.CP2K', new=FakeCP2K):
        db_path = Path(tmpdir) / 'data.db'
        db_path.unlink(missing_ok=True)
        sim = ASESimulator(scratch_dir=tmpdir, ase_db_path=str(db_path), clean_after_run=False)
        out_res, traj_res, extra = sim.optimize_structure('name', strc, config_name, charge=1)
        assert out_res.energy < traj_res[0].energy

        # Find the output directory
        run_dir = next(Path(tmpdir).glob('name/opt_*'))
        assert run_dir.is_dir()

        # Make sure everything is stored in the DB
        with connect(db_path) as db:
            assert len(db) <= len(traj_res)
            assert next(db.select())['total_charge'] == 1

        # Make sure it doesn't write new stuff
        sim.optimize_structure('name', strc, config_name, charge=1)
        with connect(db_path) as db:
            assert len(db) <= len(traj_res) + 2  # Some have same geometry, different cell -> different record
            assert next(db.select())['total_charge'] == 1

        # Make sure it can deal with a bad restart file
        (run_dir / 'opt.traj').write_text('bad')  # Kill the restart file
        sim.optimize_structure('name', strc, config_name, charge=1)
        with connect(db_path) as db:
            assert len(db) <= len(traj_res) + 2
            assert next(db.select())['total_charge'] == 1

        # Make sure it cleans up after itself
        sim.clean_after_run = True
        shutil.rmtree(run_dir)
        sim.optimize_structure('name', strc, config_name, charge=1)
        with connect(db_path) as db:
            assert len(db) <= len(traj_res) + 2
            assert next(db.select())['total_charge'] == 1
        assert not run_dir.is_dir()


def test_solvent(strc, tmpdir):
    """Test running computations with a solvent"""

    # Run a test with a patched executor
    db_path = str(tmpdir / 'data.db')
    sim = ASESimulator(scratch_dir=tmpdir, ase_db_path=db_path, clean_after_run=True)
    config = sim.create_configuration('cp2k_blyp_szv', strc, solvent='acn', charge=0)
    assert 'ALPHA' in config['kwargs']['inp']

    # Make sure there are no directories left
    assert len(list(Path(tmpdir).glob('ase_*'))) == 0

    # Run the calculation
    result, metadata = sim.compute_energy('name', strc, 'cp2k_blyp_szv', charge=0, solvent='acn')
    assert result.energy

    # Make sure the run directory was deleted
    assert len(list(Path(tmpdir).glob('name/single_*'))) == 0

    # Check that the data was added
    with connect(db_path) as db:
        assert db.count() == 1


@mark.parametrize('config_name', ['mopac_pm7'] + (['xtb'] if has_xtb else []))
def test_fast_methods(tmpdir, config_name):
    strc = write_to_string(molecule('CH4'), 'xyz')
    sim = ASESimulator(scratch_dir=tmpdir)

    # Ensure we get a different single point energy
    neutral, _ = sim.compute_energy('name', strc, config_name=config_name, charge=0)
    charged, _ = sim.compute_energy('name', strc, config_name=config_name, charge=1)
    solvated, _ = sim.compute_energy('name', strc, config_name=config_name, charge=0, solvent='acn')
    assert neutral.energy != charged.energy
    assert neutral.energy != solvated.energy

    # Ensure it relaxes under charge
    charged_opt, _, _ = sim.optimize_structure('name', strc, config_name=config_name, charge=-1)
    charged_opt_neutral, _ = sim.compute_energy('name', charged_opt.xyz, config_name=config_name, charge=0)
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


def test_gaussian_opt(tmpdir):
    strc = write_to_string(molecule('H2O'), 'xyz')
    sim = ASESimulator(gaussian_command='g16', scratch_dir=tmpdir)

    if shutil.which('g16') is None:
        # Fake execution by having it copy a known output to a target directory
        sim.gaussian_command = f'cp {(_files_dir / "Gaussian-relax.log").absolute()} Gaussian.log'

    relaxed, traj, _ = sim.optimize_structure('name', strc, 'gaussian_b3lyp_6-31g(2df,p)', charge=0)
    assert relaxed.energy < traj[0].energy
    assert relaxed.forces.max() < 0.01


def test_opt_failure(tmpdir):
    atoms = molecule('bicyclobutane')
    strc = write_to_string(atoms, 'xyz')
    sim = ASESimulator(scratch_dir=tmpdir, optimization_steps=1)

    # Run with fewer steps than needed
    with raises(ValueError):
        opt, steps, _ = sim.optimize_structure('test', strc, 'mopac_pm7', charge=0, solvent=None)

    # Run with enough
    sim.optimization_steps = 100
    opt, steps, _ = sim.optimize_structure('test', strc, 'mopac_pm7', charge=0, solvent=None)
    assert np.max(opt.forces) <= 0.02, 'Optimization did not finish successfully'
    assert len(steps) < 100
