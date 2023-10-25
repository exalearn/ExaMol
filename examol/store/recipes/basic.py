"""Properties computed for many types of applications"""

import numpy as np
from ase import units

from examol.store.models import MoleculeRecord
from examol.store.recipes.base import PropertyRecipe, RequiredGeometry, RequiredEnergy


class SolvationEnergy(PropertyRecipe):
    """Compute the solvation energy in kcal/mol

    Args:
        config_name: Name of the configuration used to compute energy
        solvent: Target solvent
    """

    def __init__(self, config_name: str, solvent: str):
        super().__init__('solvation_energy', level=f'{config_name}-{solvent}')
        self.solvent = solvent
        self.config_name = config_name

    @classmethod
    def from_name(cls, name: str, level: str) -> 'SolvationEnergy':
        config_name, solvent = level.split("-")
        return cls(config_name, solvent)

    @property
    def recipe(self) -> dict[RequiredGeometry, list[RequiredEnergy]]:
        return {
            RequiredGeometry(config_name=self.config_name, charge=0): [RequiredEnergy(config_name=self.config_name, charge=0, solvent=self.solvent)]
        }

    def compute_property(self, record: MoleculeRecord) -> float:
        # Get the lowest-energy conformer with both solvent and solvation energy
        vacuum_energy: float = np.inf
        output: float | None = None
        for conf in record.conformers:
            vac_ind = conf.get_energy_index(self.config_name, 0, None)
            sol_ind = conf.get_energy_index(self.config_name, 0, self.solvent)
            if (vac_ind is not None) and (sol_ind is not None) and conf.energies[vac_ind].energy < vacuum_energy:
                output = conf.energies[sol_ind].energy - conf.energies[vac_ind].energy
                vacuum_energy = conf.energies[sol_ind].energy

        if output is None:
            raise ValueError(f'Missing data for config="{self.config_name}", solvent={self.solvent}')
        return output * units.mol / units.kcal
