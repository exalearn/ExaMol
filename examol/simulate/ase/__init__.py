"""Utilities for simulation using ASE"""
import json
import os
from hashlib import sha512
from pathlib import Path
from shutil import rmtree, move
from time import perf_counter
from typing import Any

import ase
from ase import units
from ase.calculators.cp2k import CP2K
from ase.db import connect
from ase.io import Trajectory
from ase.io.ulm import InvalidULMFileError
from ase.optimize import LBFGSLineSearch

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


class ASESimulator(BaseSimulator):
    """Use ASE to perform simulations"""

    def __init__(self,
                 cp2k_command: str | None = None,
                 cp2k_buffer: float = 6.0,
                 scratch_dir: Path | str | None = None,
                 clean_after_run: bool = True,
                 ase_db_path: Path | str | None = None):
        """

        Args:
            cp2k_command: Command to launch CP2K
            cp2k_buffer: Length of buffer to place around molecule for CP2K (units: Ang)
            scratch_dir: Path in which to create temporary directories
            clean_after_run: Whether to clean output files after a run exits successfully
            ase_db_path: Path to an ASE db in which to store results
        """
        super().__init__(scratch_dir)
        self.cp2k_command = 'cp2k_shell' if cp2k_command is None else cp2k_command
        self.cp2k_buffer = cp2k_buffer
        self.ase_db_path = None if ase_db_path is None else Path(ase_db_path).absolute()
        self.clean_after_run = clean_after_run

    def create_configuration(self, name: str, charge: int, solvent: str | None, **kwargs) -> Any:
        if name.startswith('cp2k_blyp'):
            # Get the name the basis set
            basis_set_id = name.rsplit('_')[-1]
            basis_set_name = f'{basis_set_id}-GTH'.upper()

            # Get the cutoff
            assert basis_set_name in _cutoff_lookup, f'Cutoff energy not defined for {basis_set_name}'
            cutoff = _cutoff_lookup[basis_set_name]

            assert solvent is None, 'We do not yet support solvents'
            return {
                'name': 'cp2k',
                'kwargs': dict(
                    xc=None,
                    charge=charge,
                    uks=charge != 0,
                    inp="""&FORCE_EVAL
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
""",
                    cutoff=cutoff * units.Ry,
                    max_scf=10,
                    basis_set_file='GTH_BASIS_SETS',
                    basis_set=basis_set_name,
                    pseudo_potential='GTH-BLYP',
                    poisson_solver=None,
                    command=self.cp2k_command)
            }

    def optimize_structure(self, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, **kwargs) \
            -> tuple[SimResult, list[SimResult], str | None]:
        start_time = perf_counter()  # Measure when we started

        # Make the configuration
        calc_cfg = self.create_configuration(config_name, charge, solvent)

        # Parse the XYZ file into atoms
        atoms = utils.read_from_string(xyz, 'xyz')

        # Make the run directory based on a hash of the input configuration
        hasher = sha512()
        hasher.update(xyz.encode())
        hasher.update(config_name.encode())
        hasher.update(str(charge).encode())
        if solvent is not None:
            hasher.update(hasher)
        run_path = self.scratch_dir / Path(f'ase_opt_{hasher.hexdigest()[:8]}')
        run_path.mkdir(exist_ok=True, parents=True)

        # Run inside a temporary directory
        old_path = Path.cwd()
        try:
            os.chdir(run_path)
            with utils.make_ephemeral_calculator(calc_cfg) as calc:
                # Buffer the cell if using CP2K
                if isinstance(calc, CP2K):
                    utils.buffer_cell(atoms)

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
            out_strc = utils.write_to_string(atoms, 'xyz')
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
                    traj_xyz = utils.write_to_string(atoms, 'xyz')
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
                hasher.update(atoms.positions.round(3).tobytes())
                hasher.update(atoms.get_chemical_formula(mode='all', empirical=False).encode('ascii'))
                atoms_hash = hasher.hexdigest()[-16:] + "="

                # See if the database already has this record
                if db.count(atoms_hash=atoms_hash, config_name=config_name, total_charge=charge, solvent=str(solvent)) > 0:
                    continue

                db.write(atoms, atoms_hash=atoms_hash, config_name=config_name, total_charge=charge, solvent=str(solvent))
