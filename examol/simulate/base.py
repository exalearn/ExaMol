"""Base class defining the interfaces for common simulation operations"""
from dataclasses import dataclass
from pathlib import Path
from io import StringIO
from typing import Any

import ase
import numpy as np
from ase.io.xyz import read_xyz
from six import StringIO

from examol.utils.conversions import read_from_string


@dataclass
class SimResult:
    """Stores the results from a calculation in a code-agnostic format"""
    # Information about the result
    config_name: str = ...  # Name of the configuration
    charge: int = ...  # Charge of the molecule
    solvent: str | None = ...  # Solvent around the molecule, if any

    # Outputs
    xyz: str = ...  # 3D geometry of the molecule
    energy: float | None = None  # Energy of the molecule (units: eV)
    forces: np.ndarray | None = None  # Forces acting on each atom  (units: eV/Ang)

    @property
    def atoms(self) -> ase.Atoms:
        """ASE Atoms object of """
        return read_from_string(self.xyz, 'xyz')


class BaseSimulator:
    """Interface for tools that perform common operations"""

    def __init__(self, scratch_dir: Path | str | None):
        """
        Args:
            scratch_dir: Path in which to create temporary directories
        """
        self.scratch_dir: Path | None = None if scratch_dir is None else Path(scratch_dir)

    def create_configuration(self, name: str, charge: int, solvent: str | None, **kwargs) -> Any:
        """Create the configuration needed for a certain computation

        Args:
            name: Name of the computational method
            charge: Charge on the system
            solvent: Name of any solvent
        """
        ...

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
        ...

    def compute_energy(self, xyz: str, config_name: str, forces: bool = True,
                       charge: int = 0, solvent: str | None = None, **kwargs) -> tuple[SimResult, str | None]:
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
        ...
