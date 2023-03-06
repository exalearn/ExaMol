"""Tools for computing the properties of molecules from their record"""
from ase import units
import numpy as np

from examol.store.models import MoleculeRecord


class PropertyRecipe:
    """Compute the property given """

    def __init__(self, name: str, level: str):
        """
        Args:
            name: Name of the property
            level: Level at which it is computed
        """
        self.name = name
        self.level = level

    def update_record(self, record: MoleculeRecord):
        """Compute a property and update the record

        Args:
            record: Record to be updated
        """

        value = self.compute_property(record)
        if self.name not in record.properties:
            record.properties[self.name] = dict()
        record.properties[self.name][self.level] = value

    def compute_property(self, record: MoleculeRecord) -> float:
        """Compute the property

        Args:
            record: Data about the molecule

        Returns:
            Property value
        """
        ...


class SolvationEnergy(PropertyRecipe):
    """Compute the solvation energy in kcal/mol"""

    def __init__(self, config_name: str, solvent: str):
        """
        Args:
            config_name: Name of the configuration used to compute energy
            solvent: Target solvent
        """
        super().__init__('solvation_energy', level=f'{config_name}_{solvent}')
        self.solvent = solvent
        self.config_name = config_name

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

        assert output is not None, f'Missing data for config="{self.config_name}", solvent={self.solvent}'
        return output * units.mol / units.kcal
