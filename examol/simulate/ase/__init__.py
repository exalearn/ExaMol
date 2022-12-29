"""Utilities for simulation using ASE"""
import os
from hashlib import sha512
from pathlib import Path
from shutil import rmtree, copy
from typing import Any

from ase import units
from ase.calculators.cp2k import CP2K
from ase.io import Trajectory
from ase.optimize import LBFGS

from . import utils
from ..base import BaseSimulator, SimResult


class ASESimulator(BaseSimulator):
    """Use ASE to perform simulations"""

    def __init__(self,
                 cp2k_command: str | None = None,
                 cp2k_buffer: float = 6.0,
                 scratch_dir: str | None = None):
        """

        Args:
            cp2k_command: Command to launch CP2K
            cp2k_buffer: Length of buffer to place around molecule for CP2K (units: Ang)
            scratch_dir: Path in which to create temporary directories
        """
        super().__init__(scratch_dir)
        self.cp2k_command = 'cp2k_shell' if cp2k_command is None else cp2k_command
        self.cp2k_buffer = cp2k_buffer

    def create_configuration(self, name: str, charge: int, solvent: str | None, **kwargs) -> Any:
        if name.startswith('cp2k_blyp'):
            # Get the name the basis set
            basis_set_id = name.rsplit('_')[-1]
            basis_set_name = f'{basis_set_id}-GTH'.upper()

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
     MAX_SCF 5
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
                    cutoff=600 * units.Ry,
                    max_scf=10,
                    basis_set_file='GTH_BASIS_SETS',
                    basis_set=basis_set_name,
                    pseudo_potential='GTH-BLYP',
                    poisson_solver=None,
                    command=self.cp2k_command)
            }

    def optimize_structure(self, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, **kwargs) \
            -> (SimResult, list[SimResult], str | None):
        # Make the configuration
        calc_cfg = self.create_configuration(config_name, charge, solvent)

        # Parse the XYZ file into atoms
        atoms = utils.read_from_string(xyz, 'xyz')

        # Make the run directory based on a hash of the input configuration
        hasher = sha512()
        hasher.update(xyz.encode('ascii'))
        hasher.update(config_name.encode())
        hasher.update(bytes(charge))
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

                # Attach the calculator
                atoms.set_calculator(calc)

                # Save the history in a separate file, if stored
                traj_path = Path('lbfgs.traj')
                if traj_path.is_file():
                    copy(traj_path, 'history.traj')

                # Make the optimizer
                dyn = LBFGS(atoms, logfile='opt.log', trajectory=str(traj_path))

                # Reply the trajectory
                if Path('history.traj').is_file():
                    dyn.replay_trajectory('history.traj')

                # Run an optimization
                dyn.run(fmax=0.01)

                # Gather the outputs
                #  Start with the output structure
                out_strc = utils.write_to_string(atoms, 'xyz')
                out_result = SimResult(config_name=config_name, charge=charge, solvent=solvent,
                                       xyz=out_strc, energy=atoms.get_potential_energy(), forces=atoms.get_forces())

                # Get the trajectory
                out_traj = []
                with Trajectory(str(traj_path), mode='r') as traj:
                    for atoms in traj:
                        traj_xyz = utils.write_to_string(atoms, 'xyz')
                        traj_res = SimResult(config_name=config_name, charge=charge, solvent=solvent,
                                             xyz=traj_xyz, energy=atoms.get_potential_energy(), forces=atoms.get_forces())
                        out_traj.append(traj_res)

                # Read in the output log
                out_log = Path('opt.log').read_text()

                return out_result, out_traj, out_log

        finally:
            os.chdir(old_path)
            rmtree(run_path)
