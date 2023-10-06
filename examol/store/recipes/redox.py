"""Properties specific to electrochemical applications"""
from examol.store.models import MoleculeRecord
from examol.store.recipes.base import PropertyRecipe, RequiredGeometry, RequiredEnergy


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
