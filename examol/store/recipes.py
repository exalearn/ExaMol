"""Tools for computing the properties of molecules from their record"""
from dataclasses import dataclass, field

from ase import units
import numpy as np

from examol.simulate.initialize import add_initial_conformer
from examol.store.models import MoleculeRecord


@dataclass(frozen=True)
class SimulationRequest:
    """Request for a specific simulation type"""

    xyz: str = field(repr=False)
    """XYZ structure to use as the starting point"""
    optimize: bool = ...
    """Whether to perform an optimization"""
    config_name: str = ...
    """Name of the computation configuration"""
    charge: int = ...
    """Charge on the molecule"""
    solvent: str | None = ...
    """Name of solvent, if any"""


@dataclass(frozen=True)
class RequiredGeometry:
    """Geometry level required for a recipe"""

    config_name: str = ...
    """Level of computation required for this geometry"""
    charge: int = ...
    """Charge on the molecule used during optimization"""


@dataclass(frozen=True)
class RequiredEnergy:
    """Energy computation level required for a geometry"""

    config_name: str = ...
    """Level of computation required for the energy"""
    charge: int = ...
    """Charge on the molecule"""
    solvent: str | None = None
    """Name of solvent, if any"""


class PropertyRecipe:
    """Compute the property given a :class:`~examol.store.models.MoleculeRecord`

    **Creating a New Recipe**

    Define a recipe by implementing three operations:

    1. :meth:`__init__`: Take a users options for the recipe (e.g., what level of accuracy to use)
        then define a name and level for the recipe. Pass the name and level to the superclass's constructor.
        It is better to avoid using underscores when creating the name as underscores are
        used in the names of simulation configurations.
    2. :meth:`recipe`: Return a mapping of the different types of geometries defined
        using :class:`RequiredGeometry` and the energies which must be computed for
        each geometry using :class:`RequiredEnergy`.
    3. :meth:`compute_property`: Compute the property using the record and raise
        either a ``ValueError``, ``KeyError``, or ``AssertionError`` if the record
        lacks the required information.
    4. :meth:`from_name`: Restore a recipe from its name and level.
    """

    def __init__(self, name: str, level: str):
        """
        Args:
            name: Name of the property
            level: Level at which it is computed
        """
        self.name = name
        self.level = level

    @classmethod
    def from_name(cls, name: str, level: str) -> 'PropertyRecipe':
        """Generate a recipe from the name

        Args:
            name: Name of the property
            level: Level at which it is computed
        """
        raise NotImplementedError()

    @property
    def recipe(self) -> dict[RequiredGeometry, list[RequiredEnergy]]:
        """List of the geometries required for this recipe and the energies which must be computed for them"""
        raise NotImplementedError()

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

    def suggest_computations(self, record: MoleculeRecord) -> list[SimulationRequest]:
        """Generate a list of computations that should be performed next on a molecule

        The list of computations may not be sufficient to complete a recipe.
        For example, you may need to first relax a structure and then compute the energy of the relaxed
        structure under different conditions.

        Args:
            record: Data about the molecule
        Returns:
            List of computations to perform
        """

        output = []
        for geometry, energies in self.recipe.items():
            # Determine if the geometry has been computed
            matching_conformers = [
                (x.get_energy(geometry.config_name, geometry.charge, solvent=None), x) for x in record.conformers
                if x.source == 'relaxation' and x.config_name == geometry.config_name and x.charge == geometry.charge
            ]
            if len(matching_conformers) > 0:
                _, conformer = min(matching_conformers)
            else:
                # If there are no conformers, use the most stable one from MMFF94
                if len(record.conformers) == 0:
                    add_initial_conformer(record)
                    conf, _ = record.find_lowest_conformer(config_name='mmff', charge=0, solvent=None, optimized_only=False)
                    output.append(
                        SimulationRequest(xyz=conf.xyz, config_name=geometry.config_name, charge=geometry.charge, optimize=True, solvent=None)
                    )
                    continue

                # Preference conformers created using the same method followed by same charge, followed by creation date
                best_score = (True, float('inf'), float('inf'))
                best_xyz = None
                for conf in record.conformers:
                    my_score = (conf.config_name != geometry.config_name, abs(conf.charge - geometry.charge), -conf.date_created.timestamp())
                    if my_score < best_score:
                        best_score = my_score
                        best_xyz = conf.xyz

                output.append(
                    SimulationRequest(xyz=best_xyz, config_name=geometry.config_name, charge=geometry.charge, optimize=True, solvent=None)
                )
                continue

            # Add any required energies
            for energy in energies:
                if conformer.get_energy_index(energy.config_name, charge=energy.charge, solvent=energy.solvent) is None:
                    output.append(
                        SimulationRequest(xyz=conformer.xyz, config_name=energy.config_name, charge=energy.charge, optimize=False, solvent=energy.solvent)
                    )

        return output


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


class RedoxEnergy(PropertyRecipe):
    """Compute the redox energy for a molecule

    The level is named by the configuration used to compute the energy,
    whether a solvent was included, and whether we are computing the vertical or adiabatic energy.

    Args:
        charge: Amount the charge of the molecule should change by
        energy_config: Configuration used to compute the energy
        solvent: Solvent in which molecule is dissolved, if any
    """

    def __init__(self, charge: int, energy_config: str, vertical: bool = False, solvent: str | None = None):
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
            level += "-" + solvent
        level += ("-vertical" if vertical else "-adiabatic")
        super().__init__(name, level)

        # Save the settings
        self.energy_config = energy_config
        self.charge = charge
        self.vertical = vertical
        self.solvent = solvent

    @classmethod
    def from_name(cls, name: str, level: str) -> 'RedoxEnergy':
        # Determine the charge state
        charge = -1 if 'reduction' in name else 1
        if 'double' in name:
            charge *= 2

        # Determine the level
        if level.count('-') == 1:
            solvent = None
            config_name, approx = level.split("-")
        else:
            config_name, solvent, approx = level.split("-")
        return cls(charge=charge, energy_config=config_name, vertical=approx == 'vertical', solvent=solvent)

    @property
    def recipe(self) -> dict[RequiredGeometry, list[RequiredEnergy]]:
        if self.vertical:
            return {
                RequiredGeometry(config_name=self.energy_config, charge=0): [
                    RequiredEnergy(config_name=self.energy_config, charge=0, solvent=self.solvent),
                    RequiredEnergy(config_name=self.energy_config, charge=self.charge, solvent=self.solvent)
                ]
            }
        else:
            return dict(
                (RequiredGeometry(config_name=self.energy_config, charge=c),
                 [RequiredEnergy(config_name=self.energy_config, charge=c, solvent=self.solvent)])
                for c in [0, self.charge]
            )

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
            if charged_conf.xyz_hash == neutral_conf.xyz_hash:
                raise ValueError('We do not have a relaxed charged molecule')

        return charged_energy - neutral_energy
