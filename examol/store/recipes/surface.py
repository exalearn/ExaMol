"""Properties related to surface chemistry"""
from dataclasses import dataclass, field

import ase

from examol.store.recipes.base import PropertyRecipe, SimulationRequest


class SurfaceSimulationRequest(SimulationRequest):
    """Request for a simulation involving both a molecule and surface

    The :attr:`xyz` will include both the molecule and the surface.
    New fields describe the starting position
    """

    # Metadata describing how the surface was initialized
    surface_name: str = ...
    """Name of the surface, which should map to the geometry of a slab in the slab library"""
    conformer_hash: str = ...
    """Hash of the conformer used as the starting geometry for the molecule"""
    starting_atom: int = ...
    """Index of the atom which starts nearest to the surface"""
    orientation: tuple[float, float, float]
    """Rotation angles of the molecule around the center of mass starting from the original orientation"""


@dataclass()
class SurfaceSite:
    """Adsorption site for a certain molecule

    These inputs serve as inputs to :meth:`ase.build.add_adsorbate`.
    """

    name: str | None = None
    """Optional name of the surface site (e.g., bridge)"""
    height: float = ...
    """Starting height of the adsorbate above the slab"""
    coordinate: tuple[float, float, float] = ...
    """Position of the adsorbate within the unit cell"""
    vector: tuple[float, float, float] = ...
    """Vector normal to the surface for this site"""


@dataclass()
class SurfaceRecord:
    """Describe the surface on which molecule are adsorbed"""

    # Definition of the cell
    name: str = ...
    """Name of the surface (e.g., Cl-terminated 111 NaCl)"""
    slab: ase.Atoms = ...
    """3D geometry of the surface"""
    surface_sites: list[SurfaceSite] = field(default_factory=list)
    """Possible starting positions for the adsorbate"""

    # Properties
    energy: float = ...
    """Energy of the cell (units: eV)"""


class AdsorptionEnergy(PropertyRecipe):
    """Compute the adsorption energy of a molecule on a surface

    The slab name corresponds to the name of a JSON file in a directory of "slab information."
    Each file contains the relaxed geometry of the slab, and information about how to place adsorbates on it.
    """

    # TODO (wardlt): Be able to support >1 surfaces for different catalysis
    # TODO (wardlt): Consider how to prioritize the different required calculations across surfaces
    # TODO (wardlt): Determine how to select molecule orientation
    def __init__(self, surface_name: str, config_name: str):
        super().__init__("adsorption_energy", f"{surface_name}-config_name")
        self.surface_name = surface_name
        self.config_name = config_name
