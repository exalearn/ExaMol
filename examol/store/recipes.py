"""Tools for computing the properties of molecules from their record"""
from ase import units
import numpy as np

from examol.store.models import MoleculeRecord


class PropertyRecipe:
    """Compute the property given a :class:`~MoleculeRecord`"""

    def __init__(self, name: str, level: str):
        """
        Args:
            name: Name of the property
            level: Level at which it is computed
        """
        self.name = name
        self.level = level

    def lookup(self, record: MoleculeRecord, recompute: bool = False) -> float | None:
        """Lookup the value of a property from a record

        Args:
            record: Record to be evaluated
            recompute: Whether we should attempt to recompute the property beforehand
        Returns:
            Value of the property, if available, or ``None`` if not
        """
        # Recompute, if desired
        if recompute:
            try:
                return self.update_record(record)
            except (ValueError, AssertionError, KeyError):
                return None

        # If not, look it up
        if self.name in record.properties and self.level in record.properties[self.name]:
            return record.properties[self.name][self.level]
        return None

    def update_record(self, record: MoleculeRecord) -> float:
        """Compute a property and update the record

        Args:
            record: Record to be updated
        Returns:
            Value of the property being computed
        """
        value = self.compute_property(record)
        if self.name not in record.properties:
            record.properties[self.name] = dict()
        record.properties[self.name][self.level] = value
        return value

    def compute_property(self, record: MoleculeRecord) -> float:
        """Compute the property

        Args:
            record: Data about the molecule

        Returns:
            Property value
        """
        raise NotImplementedError()


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


class RedoxEnergy(PropertyRecipe):
    """Compute the redox energy for a molecule

    The level is named by the configuration used to compute the energy,
    whether a solvent was included, and whether we are computing the vertical or adiabatic energy.

    At present, we do not check how geometries were acquired and simplify find the lowest-energy
    conformer for the neutral and, if we are computing adiabatic energies, the charge state.
    """

    def __init__(self, charge: int, energy_config: str, vertical: bool = False, solvent: str | None = None):
        """

        Args:
            charge: Amount the charge of the molecule should change by
            energy_config: Configuration used to compute the energy
            solvent: Solvent in which molecule is dissolved, if any
        """

        # Make the name of the property based on the charge state
        assert abs(charge) > 0
        prefix = {
            1: '',
            2: 'double_'
        }[abs(charge)]
        name = prefix + ('reduction_potential' if charge < 0 else 'oxidation_potential')

        # Name the level based on the energy level and solvent
        level = energy_config
        if solvent is not None:
            level += "_" + solvent
        level += ("_vertical" if vertical else "_adiabatic")
        super().__init__(name, level)

        # Save the settings
        self.energy_config = energy_config
        self.charge = charge
        self.vertical = vertical
        self.solvent = solvent

    def compute_property(self, record: MoleculeRecord) -> float:
        # Get the lowest energy conformer of the neutral molecule
        neutral_conf, neutral_energy = record.find_lowest_conformer(self.energy_config, 0, self.solvent)

        # Get the charged energy
        if self.vertical:
            # Get the energy for this conformer
            charged_energy = neutral_conf.get_energy(self.energy_config, self.charge, self.solvent)
        else:
            # Ge the lowest-energy conformer for the charged molecule
            charged_conf, charged_energy = record.find_lowest_conformer(self.energy_config, self.charge, self.solvent)
            assert charged_conf.xyz_hash != neutral_conf.xyz_hash, 'We do not have a relaxed charged molecule'

        return charged_energy - neutral_energy
