"""Utilities for simulation using ASE"""
import json
import os
from hashlib import sha512
from pathlib import Path
from shutil import rmtree, move
from time import perf_counter

import ase
from ase import units
from ase.db import connect
from ase.io import Trajectory
from ase.io.ulm import InvalidULMFileError
from ase.optimize import LBFGSLineSearch
from ase.calculators.gaussian import Gaussian

import examol.utils.conversions
from . import utils
from ..base import BaseSimulator, SimResult

# Mapping between basis set and a converged cutoff energy
#  See methods in: https://github.com/exalearn/quantum-chemistry-on-polaris/blob/main/cp2k/mt/converge-parameters-mt.ipynb
#  We increase the cutoff slightly to be on the safe side
_cutoff_lookup = {
    'TZVP-GTH': 850.,
    'DZVP-GTH': 600.,
    'SZV-GTH': 600.
}

# Base input file
_cp2k_inp = """&FORCE_EVAL
&DFT
  &XC
     &XC_FUNCTIONAL BLYP
     &END XC_FUNCTIONAL
  &END XC
  &POISSON
     PERIODIC NONE
     PSOLVER MT
  &END POISSON
  &SCF
    &OUTER_SCF
     MAX_SCF 9
    &END OUTER_SCF
    &OT T
      PRECONDITIONER FULL_ALL
    &END OT
  &END SCF
&END DFT
&SUBSYS
  &TOPOLOGY
    &CENTER_COORDINATES
    &END
  &END
&END FORCE_EVAL
"""

# Solvent data (solvent name -> (gamma, e0) for CP2K, solvent name - xTB/G name for xTB/G)
_solv_data = {
    'acn': (
        29.4500,  # http://www.ddbst.com/en/EED/PCP/SFT_C3.php
        37.5  # https://depts.washington.edu/eooptic/linkfiles/dielectric_chart%5B1%5D.pdf
    )
}
_xtb_solv_names = {'acn': 'acetonitrile'}
_gaussian_solv_names = {'acn': 'acetonitrile'}


def _compute_run_hash(charge: int, config_name: str, solvent: str | None, xyz: str) -> str:
    """Generate a summary hash for a calculation

    Args:
        charge: Charge of the cell
        config_name: Name of the configuration
        solvent: Name of the solvent, if any
        xyz: XYZ coordinates for the atoms
    Returns:
        Hash of the above contents
    """
    hasher = sha512()
    hasher.update(xyz.encode())
    hasher.update(config_name.encode())
    hasher.update(str(charge).encode())
    if solvent is not None:
        hasher.update(solvent.encode())
    run_hash = hasher.hexdigest()[:8]
    return run_hash


class ASESimulator(BaseSimulator):
    """Use ASE to perform quantum chemistry calculations

    The calculator supports calculations with the following codes:

    - *XTB*: Tight binding using the GFN2-xTB parameterization
    - *Gaussian*: Supports any of the methods and basis sets of Gaussian
      using names of the format ``gaussian_[method]_[basis]``. Supply
      additional arguments to Gaussian as keyword arguments.
    - *CP2K*: Supports only a few combinations of basis sets and XC functions,
      those for which we have determined appropriate cutoff energies:
      ``cp2k_blyp_szv``, ``cp2k_blyp_dzvp``, ``cp2k_blyp_tzvp``


    Args:
        cp2k_command: Command to launch CP2K
        gaussian_command: Command to launch Gaussian. Only the path to the executable is generally needed
        scratch_dir: Path in which to create temporary directories
        clean_after_run: Whether to clean output files after a run exits successfully
        ase_db_path: Path to an ASE db in which to store results
    """
    def __init__(self,
                 cp2k_command: str | None = None,
                 gaussian_command: str | None = None,
                 scratch_dir: Path | str | None = None,
                 clean_after_run: bool = True,
                 ase_db_path: Path | str | None = None):
        super().__init__(scratch_dir)
        self.cp2k_command = 'cp2k_shell' if cp2k_command is None else cp2k_command
        self.gaussian_command = Gaussian.command if gaussian_command is None else f'{gaussian_command} < PREFIX.com > PREFIX.log'
        self.ase_db_path = None if ase_db_path is None else Path(ase_db_path).absolute()
        self.clean_after_run = clean_after_run

    def create_configuration(self, name: str, charge: int, solvent: str | None, **kwargs) -> dict:
        if name == 'xtb':
            kwargs = {}
            if solvent is not None:
                assert solvent in _xtb_solv_names
                kwargs['solvent'] = _xtb_solv_names[solvent]
            return {'name': 'xtb', 'kwargs': kwargs, 'accuracy': 0.05}
        elif name.startswith('gaussian_'):
            # Unpack the name
            if name.count("_") != 2:
                raise ValueError('Detected the wrong number of separators. Names for the XC function and basis set should not include underscores.')
            _, xc, basis = name.split("_")

            # Create additional options
            add_options = {}
            if solvent is not None:
                add_options['SCRF'] = f'PCM,Solvent={_gaussian_solv_names.get(solvent, solvent)}'
            add_options['scf'] = 'xqc,MaxConventional=200'

            # Build the specification
            return {
                'name': 'gaussian',
                'kwargs': {
                    'command': self.gaussian_command,
                    'chk': 'gauss.chk',
                    'basis': basis,
                    'method': xc,
                    'charge': charge,
                    'mult': abs(charge) + 1,  # Assume the worst
                    **add_options,
                    **kwargs
                }
            }
        elif name.startswith('cp2k_blyp'):
            # Get the name the basis set
            basis_set_id = name.rsplit('_')[-1]
            basis_set_name = f'{basis_set_id}-GTH'.upper()

            # Get the cutoff
            assert basis_set_name in _cutoff_lookup, f'Cutoff energy not defined for {basis_set_name}'
            cutoff = _cutoff_lookup[basis_set_name]

            # Add solvent information, if desired
            inp = _cp2k_inp
            if solvent is not None:
                assert solvent in _solv_data, f"Solvent {solvent} not defined. Available: {', '.join(_solv_data.keys())}"
                gamma, e0 = _solv_data[solvent]
                # Inject it in the input file
                #  We use beta=0 and alpha+gamma=0 as these do not matter for solvation energy: https://groups.google.com/g/cp2k/c/7oYTqSIyIqI/m/7D62tXIzBgAJ
                inp = inp.replace(
                    '&END SCF\n',
                    f"""&END SCF
&SCCS
  ALPHA {-gamma}
  BETA 0
  GAMMA {gamma}
RELATIVE_PERMITTIVITY {e0}
DERIVATIVE_METHOD CD3
METHOD ANDREUSSI
&END SCCS\n""")

            return {
                'name': 'cp2k',
                'buffer_size': 10.0,
                'kwargs': dict(
                    xc=None,
                    charge=charge,
                    uks=charge != 0,
                    inp=inp,
                    cutoff=cutoff * units.Ry,
                    max_scf=10,
                    basis_set_file='GTH_BASIS_SETS',
                    basis_set=basis_set_name,
                    pseudo_potential='GTH-BLYP',
                    poisson_solver=None,
                    stress_tensor=False,
                    command=self.cp2k_command)
            }

    def optimize_structure(self, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, **kwargs) \
            -> tuple[SimResult, list[SimResult], str | None]:
        start_time = perf_counter()  # Measure when we started

        # Make the configuration
        calc_cfg = self.create_configuration(config_name, charge, solvent)

        # Parse the XYZ file into atoms
        atoms = examol.utils.conversions.read_from_string(xyz, 'xyz')

        # Make the run directory based on a hash of the input configuration
        run_hash = _compute_run_hash(charge, config_name, solvent, xyz)
        run_path = self.scratch_dir / Path(f'ase_opt_{run_hash}')
        run_path.mkdir(exist_ok=True, parents=True)

        # Run inside a temporary directory
        old_path = Path.cwd()
        try:
            os.chdir(run_path)
            with utils.make_ephemeral_calculator(calc_cfg) as calc:
                # Buffer the cell if using CP2K
                self._prepare_atoms(atoms, charge, calc_cfg)

                # Recover the history from a previous run
                traj_path = Path('lbfgs.traj')
                if traj_path.is_file():
                    try:
                        # Overwrite our atoms with th last in the trajectory
                        with Trajectory(traj_path, mode='r') as traj:
                            for atoms in traj:
                                pass

                        # Move the history so we can use it to over
                        move(traj_path, 'history.traj')
                    except InvalidULMFileError:
                        pass

                # Attach the calculator
                atoms.calc = calc

                # Make the optimizer
                dyn = LBFGSLineSearch(atoms, logfile='opt.log', trajectory=str(traj_path))

                # Reply the trajectory
                if Path('history.traj').is_file():
                    dyn.replay_trajectory('history.traj')
                    os.unlink('history.traj')

                # Run an optimization
                dyn.run(fmax=0.02, steps=250)

            # Gather the outputs
            #  Start with the output structure
            if self.ase_db_path is not None:
                self.update_database([atoms], config_name, charge, solvent)
            out_strc = examol.utils.conversions.write_to_string(atoms, 'xyz')
            out_result = SimResult(config_name=config_name, charge=charge, solvent=solvent,
                                   xyz=out_strc, energy=atoms.get_potential_energy(), forces=atoms.get_forces())

            # Get the trajectory
            out_traj = []
            with Trajectory(str(traj_path), mode='r') as traj:
                # Get all atoms in the trajectory
                traj_lst = [a for a in traj]
                if self.ase_db_path is not None:
                    self.update_database(traj_lst, config_name, charge, solvent)

                # Convert them to the output format
                for atoms in traj_lst:
                    traj_xyz = examol.utils.conversions.write_to_string(atoms, 'xyz')
                    traj_res = SimResult(config_name=config_name, charge=charge, solvent=solvent,
                                         xyz=traj_xyz, energy=atoms.get_potential_energy(), forces=atoms.get_forces())
                    out_traj.append(traj_res)

            # Read in the output log
            out_log = Path('opt.log').read_text()

            # Delete the run directory
            if self.clean_after_run:
                os.chdir(old_path)
                rmtree(run_path)

            return out_result, out_traj, json.dumps({'runtime': perf_counter() - start_time, 'out_log': out_log})

        finally:
            # Make sure we end back where we started
            os.chdir(old_path)

    def _prepare_atoms(self, atoms: ase.Atoms, charge: int, config: dict):
        """Make the atoms object ready for the simulation

        Args:
            atoms: Atoms object to be adjusted
            charge: Charge on the system
            config: Configuration detail
        """
        if 'cp2k' in config['name']:
            atoms.center(vacuum=config['buffer_size'])
        elif 'xtb' in config['name']:
            utils.initialize_charges(atoms, charge)

    def compute_energy(self, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, forces: bool = True,
                       **kwargs) -> tuple[SimResult, str | None]:
        # Make the configuration
        start_time = perf_counter()  # Measure when we started

        # Make the configuration
        calc_cfg = self.create_configuration(config_name, charge, solvent)

        # Make the run directory based on a hash of the input configuration
        run_hash = _compute_run_hash(charge, config_name, solvent, xyz)
        run_path = self.scratch_dir / Path(f'ase_single_{run_hash}')
        run_path.mkdir(exist_ok=True, parents=True)

        # Parse the XYZ file into atoms
        atoms = examol.utils.conversions.read_from_string(xyz, 'xyz')

        # Run inside a temporary directory
        old_path = Path.cwd()
        try:
            os.chdir(run_path)

            # Prepare to run the cell
            with utils.make_ephemeral_calculator(calc_cfg) as calc:
                # Make any changes to cell needed by the calculator
                self._prepare_atoms(atoms, charge, calc_cfg)

                # Run a single point
                atoms.calc = calc
                forces = atoms.get_forces() if forces else None
                energy = atoms.get_potential_energy()

                # Report the results
                if self.ase_db_path is not None:
                    self.update_database([atoms], config_name, charge, solvent)
                out_strc = examol.utils.conversions.write_to_string(atoms, 'xyz')
                out_result = SimResult(config_name=config_name, charge=charge, solvent=solvent,
                                       xyz=out_strc, energy=energy, forces=forces)

                # Delete the run directory
                if self.clean_after_run:
                    os.chdir(old_path)
                    rmtree(run_path)

                return out_result, json.dumps({'runtime': perf_counter() - start_time})

        finally:
            os.chdir(old_path)

    def update_database(self, atoms_to_write: list[ase.Atoms], config_name: str, charge: int, solvent: str | None):
        """Update the ASE database collected along with this class

        Args:
            atoms_to_write: List of Atoms objects to store in DB
            config_name: Name of the configuration used to compute energies
            charge: Charge on the system
            solvent: Name of solvent, if any
        """

        # Connect to the database
        with connect(self.ase_db_path, append=True) as db:
            for atoms in atoms_to_write:
                # Get the atom hash
                hasher = sha512()
                hasher.update(atoms.positions.round(5).tobytes())
                hasher.update(atoms.get_chemical_formula(mode='all', empirical=False).encode('ascii'))
                atoms_hash = hasher.hexdigest()[-16:] + "="

                # See if the database already has this record
                if db.count(atoms_hash=atoms_hash, config_name=config_name, total_charge=charge, solvent=str(solvent)) > 0:
                    continue

                db.write(atoms, atoms_hash=atoms_hash, config_name=config_name, total_charge=charge, solvent=str(solvent))
