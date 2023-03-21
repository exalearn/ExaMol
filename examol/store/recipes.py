"""Tools for computing the properties of molecules from their record"""
from dataclasses import dataclass

from ase import units
import numpy as np

from examol.simulate.initialize import generate_inchi_and_xyz
from examol.store.models import MoleculeRecord


@dataclass
class SimulationRequest:
    """Request for a specific simulation type

    Attributes:
        xyz: XYZ structure to use as the starting point
        optimize: Whether to perform an optimization
        config_name: Name of the computation
        charge: Charge on the molecule
        solvent: Name of solvent, if any
    """

    xyz: str = ...
    optimize: bool = ...
    config_name: str = ...
    charge: int = ...
    solvent: str | None = ...


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

    def suggest_computations(self, record: MoleculeRecord) -> list[SimulationRequest]:
        try:
            # Find the lowest-energy conformer
            conf, _ = record.find_lowest_conformer(self.config_name, charge=0, solvent=None)

            # If there is a structure, see if have the energy in the solvent
            energy_ind = conf.get_energy_index(self.config_name, 0, solvent=self.solvent)
            if energy_ind is None:
                return [SimulationRequest(xyz=conf.xyz, optimize=False, config_name=self.config_name, charge=0, solvent=self.solvent)]
            else:
                return []
        except ValueError:  # TODO (wardlt): Avoid using exceptions, especially such general ones. I _assume_ ValueError means no structure
            # Make a XYZ structure
            _, xyz = generate_inchi_and_xyz(record.identifier.smiles)
            return [SimulationRequest(xyz=xyz, optimize=True, config_name=self.config_name, charge=0, solvent=None)]


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

    def suggest_computations(self, record: MoleculeRecord) -> list[SimulationRequest]:
        output = []
        # Determine if we need to perform a neutral relaxation
        neutral_conf = None
        try:
            neutral_conf, _ = record.find_lowest_conformer(self.energy_config, 0, solvent=None)
        except ValueError:  # TODO (wardlt): As above, avoid exceptions
            # If fails, request a relaxation
            _, starting_xyz = generate_inchi_and_xyz(record.identifier.smiles)
            output.append(SimulationRequest(xyz=starting_xyz, optimize=True, config_name=self.energy_config, charge=0, solvent=None))

        # See if we need a charged relaxation or single point
        charged_conf = None
        if self.vertical:
            if neutral_conf is not None:
                charged_conf = neutral_conf
                if neutral_conf.get_energy_index(self.energy_config, self.charge, solvent=None) is None:
                    output.append(
                        SimulationRequest(xyz=neutral_conf.xyz, optimize=False, config_name=self.energy_config, charge=self.charge, solvent=None)
                    )
        else:
            try:
                charged_conf, _ = record.find_lowest_conformer(self.energy_config, self.charge, solvent=None)
                if charged_conf.xyz_hash == neutral_conf.xyz_hash and not self.vertical:
                    raise ValueError('We need to do a relaxation')
            except ValueError:
                if neutral_conf is None:
                    _, starting_xyz = generate_inchi_and_xyz(record.identifier.smiles)
                else:
                    starting_xyz = neutral_conf.xyz
                output.append(SimulationRequest(xyz=starting_xyz, optimize=True, config_name=self.energy_config, charge=self.charge, solvent=None))

        # Solvation computations if needed and ready
        if self.solvent is None:
            return output

        for conf, chg in zip([neutral_conf, charged_conf], [0, self.charge]):
            if conf is not None and conf.get_energy_index(self.energy_config, chg, self.solvent) is None:
                output.append(
                    SimulationRequest(xyz=conf.xyz, optimize=False, config_name=self.energy_config, charge=chg, solvent=self.solvent)
                )
        return output
