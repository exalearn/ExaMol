"""Base class defining the interfaces for common simulation operations"""
from dataclasses import dataclass, field
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


class BaseSimulator:
    """Interface for tools that perform common operations"""

    def __init__(self, scratch_dir: Path | str | None):
        """
        Args:
            scratch_dir: Path in which to create temporary directories
        """
        self.scratch_dir: Path | None = Path('tmp') if scratch_dir is None else Path(scratch_dir)

    def create_configuration(self, name: str, xyz: str, charge: int, solvent: str | None, **kwargs) -> Any:
        """Create the configuration needed for a certain computation

        Args:
            name: Name of the computational method
            xyz: Structure being evaluated in XYZ format
            charge: Charge on the system
            solvent: Name of any solvent
        """
        raise NotImplementedError()

    def optimize_structure(self, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, **kwargs) \
            -> tuple[SimResult, list[SimResult], str | None]:
        """Minimize the energy of a structure

        Args:
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

    def compute_energy(self, xyz: str, config_name: str, charge: int = 0, solvent: str | None = None, forces: bool = True,
                       **kwargs) -> tuple[SimResult, str | None]:
        """Get the energy and forces of a structure

        Args:
            Args:
            xyz: 3D geometry of the molecule
            config_name: Name of the method
            charge: Charge on the molecule
            solvent: Name of the solvent
            **kwargs: Any other arguments for the method

        Returns:
            - Energy result
            - Other metadata produced by the computation
        """
        raise NotImplementedError()
