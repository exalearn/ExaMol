"""Base class defining the interfaces for common simulation operations"""
import json
from dataclasses import dataclass, field, asdict
from hashlib import sha512
from pathlib import Path
from typing import Any

import ase
import numpy as np

from examol.utils.conversions import read_from_string, write_to_string


@dataclass()
class SimResult:
    """Stores the results from a calculation in a code-agnostic format"""
    # Information about the result
    config_name: str = field()
    """Name of the configuration used to compute the energy"""
    charge: int = field()
    """Charge of the molecule"""
    solvent: str | None = field()
    """Solvent around the molecule, if any"""

    # Outputs
    xyz: str = field(repr=False)
    """XYZ-format structure, adjusted such that the center of mass is at the origin"""
    energy: float | None = None
    """Energy of the molecule (units: eV)"""
    forces: np.ndarray | None = None
    """Forces acting on each atom  (units: eV/Ang)"""

    def __post_init__(self):
        # Ensure the XYZ is centered about zero
        atoms = read_from_string(self.xyz, 'xyz')
        atoms.center()
        self.xyz = write_to_string(atoms, 'xyz')

    @property
    def atoms(self) -> ase.Atoms:
        """ASE Atoms object representation of the structure"""
        return read_from_string(self.xyz, 'xyz')

    def json(self, **kwargs) -> str:
        """Write the record to JSON format"""
        output = asdict(self)
        if isinstance(output['forces'], np.ndarray):
            output['forces'] = output['forces'].tolist()
        return json.dumps(output, **kwargs)


class BaseSimulator:
    """Uniform interface for common types of computations

    **Creating a New Simulator**

    There are a few considerations to weigh when fulfilling the abstract methods:

    - Use underscores in the name of method configurations.

    Args:
        scratch_dir: Path in which to create temporary directories
        retain_failed: Whether to retain failed computations
    """

    def __init__(self, scratch_dir: Path | str | None, retain_failed: bool = True):
        self.scratch_dir: Path | None = Path('tmp') if scratch_dir is None else Path(scratch_dir)
        self.retain_failed = retain_failed

    def _make_run_hash(self, xyz: str, config_name: str, charge: int, solvent: str | None) -> str:
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
        hasher.update(self.__class__.__name__.encode())
        hasher.update(xyz.encode())
        hasher.update(config_name.encode())
        hasher.update(str(charge).encode())
        if solvent is not None:
            hasher.update(solvent.encode())
        run_hash = hasher.hexdigest()[:8]
        return run_hash

    def create_configuration(self, name: str, xyz: str, charge: int, solvent: str | None, **kwargs) -> Any:
        """Create the configuration needed for a certain computation

        Args:
            name: Name of the computational method
            xyz: Structure being evaluated in XYZ format
            charge: Charge on the system
            solvent: Name of any solvent
        """
        raise NotImplementedError()

    def optimize_structure(self, mol_key: str, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, **kwargs) \
            -> tuple[SimResult, list[SimResult], str | None]:
        """Minimize the energy of a structure

        Args:
            mol_key: InChI key of the molecule being evaluated
            xyz: 3D geometry of the molecule
            config_name: Name of the method
            charge: Charge on the molecule
            solvent: Name of the solvent
            **kwargs: Any other arguments for the method

        Returns:
            - The minimized structure
            - Any intermediate structures
            - Other metadata produced by the computation
        """
        raise NotImplementedError()

    def compute_energy(self, mol_key: str, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, forces: bool = True,
                       **kwargs) -> tuple[SimResult, str | None]:
        """Get the energy and forces of a structure

        Args:
            mol_key: InChI key of the molecule being evaluated
            xyz: 3D geometry of the molecule
            config_name: Name of the method
            charge: Charge on the molecule
            solvent: Name of the solvent
            forces: Whether to compute forces
            **kwargs: Any other arguments for the method

        Returns:
            - Energy result
            - Other metadata produced by the computation
        """
        raise NotImplementedError()

    def _make_run_directory(self, run_type: str, mol_key: str, xyz: str, charge: int, config_name: str, solvent: str | None) -> Path:
        """Create a run directory for the calculation

        Args:
            run_type: Type of the run to perform (e.g., "opt", "single")
            mol_key: InChI key of the molecule being evaluated
            charge: Charge of the cell
            config_name: Name of the configuration
            solvent: Name of the solvent, if any
            xyz: XYZ coordinates for the atoms
        Returns:
            Path in which to run the computation
        """
        # Make the directory
        run_hash = self._make_run_hash(xyz, config_name, charge, solvent)
        run_path = self.scratch_dir / mol_key / Path(f'{run_type}_{run_hash}')
        run_path.mkdir(parents=True, exist_ok=True)

        # Write a calculation summary to the run path
        with open(run_path / 'summary.json', 'w') as fp:
            json.dump({
                'xyz': xyz,
                'config_name': config_name,
                'charge': charge,
                'solvent': solvent
            }, fp, indent=2)

        return run_path
