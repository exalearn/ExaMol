"""Data models used for molecular data"""
from hashlib import md5
from datetime import datetime
from typing import Collection

import ase
import numpy as np
from mongoengine import Document, DynamicEmbeddedDocument, EmbeddedDocument, IntField, EmbeddedDocumentField, DateTimeField, FloatField, DictField
from mongoengine.fields import StringField, ListField
from rdkit import Chem

from examol.simulate.base import SimResult
from examol.utils.conversions import read_from_string, write_to_string


class Identifiers(DynamicEmbeddedDocument):
    """IDs known for a molecule"""

    smiles = StringField(required=True)
    """A SMILES string"""
    inchi = StringField(required=True)
    """The InChI string"""
    pubchem_id = IntField()
    """PubChem ID, if known"""


class EnergyEvaluation(EmbeddedDocument):
    """Energy of a conformer under a certain condition"""

    energy = FloatField(required=True)
    """Energy of the conformer (eV)"""
    config_name = StringField(required=True)
    """Configuration used to compute the energy"""
    charge = IntField(required=True)
    """Charge used when computing the energy"""
    solvent = StringField()
    """Solvent used, if any"""

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.config_name == other.config_name \
            and self.charge == other.charge \
            and self.solvent == other.solvent


class Conformer(EmbeddedDocument):
    """Describes a single conformer of a molecule"""

    # Define the structure
    xyz = StringField(required=True)
    """XYZ-format description of the atomic coordinates"""
    xyz_hash = StringField(required=True)
    """MDF hash of the XYZ coordinates"""

    # Provenance of the structure
    date_created = DateTimeField(required=True)
    """Date this conformer was inserted"""
    source = StringField(required=True, choices=['relaxation', 'other'])
    """Method used to generate this structure (e.g., via relaxation)"""
    config_name = StringField()
    """Configuration used to relax the structure, if applicable"""
    charge = IntField()
    """Charge used when relaxing the structure"""

    # Energies of the structure
    energies: list[EnergyEvaluation] = ListField(EmbeddedDocumentField(EnergyEvaluation))
    """List of energies for this structure"""

    @property
    def atoms(self) -> ase.Atoms:
        return read_from_string(self.xyz, 'xyz')

    @classmethod
    def from_simulation_result(cls, sim_result: SimResult, source: str = 'relaxation') -> 'Conformer':
        """Create a new object from a simulation

        Args:
            sim_result: Simulation result
            source: How this conformer was determined
        Returns:
            An initialized conformer record
        """

        # Make sure the simulation results is center
        atoms = read_from_string(sim_result.xyz, 'xyz')
        atoms.center()
        xyz = write_to_string(atoms, 'xyz')

        # Convert the sim result to a database record
        new_record = cls(
            xyz=xyz,
            xyz_hash=md5(xyz.encode()).hexdigest(),
            date_created=datetime.now(),
            source=source,
            config_name=sim_result.config_name,
            charge=sim_result.charge,
        )
        new_record.add_energy(sim_result)
        return new_record

    def add_energy(self, sim_result: SimResult) -> bool:
        """Add the energy from a simulation result

        Args:
            sim_result: Result to be added
        """

        new_energy = EnergyEvaluation(
            energy=sim_result.energy,
            config_name=sim_result.config_name,
            charge=sim_result.charge,
            solvent=sim_result.solvent
        )
        if new_energy in self.energies:
            return False
        self.energies.append(new_energy)
        return True

    def get_energy_index(self, config_name: str, charge: int, solvent: str | None) -> int | None:
        """Get the index of the record for a certain level of energy

        Args:
            config_name: Name of the compute configuration
            charge: Charge of the molecule
            solvent: Solvent in which the molecule is dissolved
        Returns:
            Index of the record, if available, or ``None``, if not.
        """

        for i, record in enumerate(self.energies):
            if record.config_name == config_name and \
                    record.charge == charge and \
                    record.solvent == solvent:
                return i
        return None

    def get_energy(self, config_name: str, charge: int, solvent: str | None) -> float:
        """Get the energy for a certain level

        Args:
            config_name: Name of the compute configuration
            charge: Charge of the molecule
            solvent: Solvent in which the molecule is dissolved
        Returns:
            Index of the record, if available, or ``None``, if not.
        """

        ind = self.get_energy_index(config_name, charge, solvent)
        if ind is None:
            raise ValueError(f'No energy available for config="{config_name}", charge={charge}, solvent={solvent}')
        return self.energies[ind].energy


class MoleculeRecord(Document):
    """Defines whatever we know about a molecule"""

    # Identifiers
    key = StringField(min_length=27, max_length=27, required=True, primary_key=True)
    """InChI key"""
    identifier: Identifiers = EmbeddedDocumentField(Identifiers, help_text='')
    """Collection of identifiers which define the molecule"""
    names = ListField(StringField())
    """Names this molecule is known by"""
    subsets = ListField(StringField())
    """List of subsets this molecule is part of"""

    # Data about the molecule
    conformers: list[Conformer] = ListField(EmbeddedDocumentField(Conformer))
    """All known conformers for this molecule"""

    # Characteristics
    properties: dict[str, dict[str, float]] = DictField()
    """Properties available for the molecule"""

    @classmethod
    def from_identifier(cls, smiles: str | None = None, inchi: str | None = None):
        assert (smiles is not None) ^ (inchi is not None), "You must supply either smiles or inchi, and not both"

        # Load in the molecule
        if smiles is None:
            mol = Chem.MolFromInchi(inchi)
        else:
            mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Molecule failed to parse: {smiles if inchi is None else inchi}')

        # Create the object
        return cls(key=Chem.MolToInchiKey(mol), identifier=Identifiers(smiles=Chem.MolToSmiles(mol), inchi=Chem.MolToInchi(mol)))

    def add_energies(self, result: SimResult, opt_steps: Collection[SimResult] = (), match_tol: float = 1e-3) -> bool:
        """Add a new set of energies to a structure

        Will add a new conformer if the structure does not yet exist

        If provided, will match the energies of any materials within the optimization steps

        Args:
            result: Energy computation to be added
            opt_steps: Optimization steps, if available
            match_tol: Maximum absolute difference between XYZ coordinates to match
        Returns:
            Whether a new conformer was added
        """

        # Prepare to match conformers
        conf_pos = [c.atoms.positions for c in self.conformers]

        def _match_conformers(atoms: ase.Atoms) -> int | None:
            atoms = atoms.copy()
            atoms.center()
            positions = atoms.positions
            for i, c in enumerate(conf_pos):
                if np.isclose(positions, c, atol=match_tol).all():
                    return i

        # First try to add optimization steps
        for step in opt_steps:
            if (match_id := _match_conformers(step.atoms)) is not None:
                self.conformers[match_id].add_energy(step)

        # Get the atomic positions for me
        my_match = _match_conformers(result.atoms)
        if my_match is None:
            self.conformers.append(Conformer.from_simulation_result(result))
            return True
        else:
            self.conformers[my_match].add_energy(result)
            return False

    def find_lowest_conformer(self, config_name: str, charge: int, solvent: str | None) -> tuple[Conformer, float]:
        """Get the energy of the lowest-energy conformer of a molecule in a certain state

        Args:
            config_name: Name of the compute configuration
            charge: Charge of the molecule
            solvent: Solvent in which the molecule is dissolved
        Returns:
            - Lowest-energy conformer
            - Energy of the structure (eV)
        """

        # Output results
        lowest_energy: float = np.inf
        stable_conformer: Conformer | None = None

        # Check all conformers
        for conf in self.conformers:
            energy_ind = conf.get_energy_index(config_name, charge, solvent)
            if energy_ind is not None and conf.energies[energy_ind].energy < lowest_energy:
                stable_conformer = conf
                lowest_energy = conf.energies[energy_ind].energy

        if stable_conformer is None:
            raise ValueError(f'No energy evaluations found for config_name="{config_name}, charge={charge}, solvent="{solvent}"')

        return stable_conformer, lowest_energy
