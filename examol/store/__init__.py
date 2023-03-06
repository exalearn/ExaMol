"""Tools related to storeing then retrieving data about molecules"""

from hashlib import md5
from datetime import datetime
from typing import Collection

import ase
import numpy as np
from mongoengine import Document, DynamicEmbeddedDocument, EmbeddedDocument, IntField, EmbeddedDocumentField, DateTimeField, FloatField
from mongoengine.fields import StringField, ListField
from rdkit import Chem

from examol.simulate.base import SimResult
from examol.utils.conversions import read_from_string


class Identifiers(DynamicEmbeddedDocument):
    """IDs known for a molecule"""

    smiles = StringField(required=True)
    inchi = StringField(required=True)
    pubchem_id = IntField()


class EnergyEvaluation(EmbeddedDocument):
    """Energy of a conformer under a certain condition"""

    energy = FloatField(required=True, help_text='Energy of the conformer (eV)')
    config_name = StringField(required=True, help_text='Configuration used to compute the energy')
    charge = IntField(required=True, help_text='Charge used when computing the energy')
    solvent = StringField(help_text='Solvent used, if any')

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        return self.config_name == other.config_name \
            and self.charge == other.charge \
            and self.solvent == other.solvent


class Conformer(EmbeddedDocument):
    """Describes a single conformer of a molecule"""

    # Define the structure
    xyz = StringField(required=True, help_text='XYZ-format description of the atomic coordinates')
    xyz_hash = StringField(required=True, help_text='MD5 hash of xyz')

    # Provenance of the structure
    date_created = DateTimeField(required=True, help_text='Date this conformer was inserted')
    source = StringField(required=True, choices=['relaxation', 'other'], help_text='Method used to generate this structure')
    config_name = StringField(help_text='Configuration used to relax the structure')
    charge = IntField(help_text='Charge used when relaxing the structure')

    # Energies of the structure
    energies: list[EnergyEvaluation] = ListField(EmbeddedDocumentField(EnergyEvaluation), help_text='List of energies for this structure')

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

        new_record = cls(
            xyz=sim_result.xyz,
            xyz_hash=md5(sim_result.xyz.encode()).hexdigest()[-32:],
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


class MoleculeRecord(Document):
    """Defines whatever we know about a molecule"""

    # Identifiers
    key = StringField(min_length=27, max_length=27, required=True, primary_key=True, help_text='InChI key')
    identifier = EmbeddedDocumentField(Identifiers, help_text='Collection of identifiers which define the molecule')
    names = ListField(StringField(), help_text='Names this molecule is known by')

    # Data about the molecule
    conformers: list[Conformer] = ListField(EmbeddedDocumentField(Conformer))

    # Characteristics
    subsets = ListField(StringField(), help_text='List of subsets this molecule is part of')

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

    def add_energies(self, result: SimResult, opt_steps: Collection[SimResult] = (), match_tol: float = 1e-2) -> bool:
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

        def _match_conformers(positions: np.ndarray) -> int | None:
            for i, c in enumerate(conf_pos):
                if np.isclose(positions, conf_pos, atol=match_tol).all():
                    return i

        # First try to add optimization steps
        for step in opt_steps:
            if (match_id := _match_conformers(step.atoms.positions)) is not None:
                self.conformers[match_id].add_energy(result)

        # Get the atomic positions for me
        my_match = _match_conformers(result.atoms.positions)
        if my_match is None:
            self.conformers.append(Conformer.from_simulation_result(result))
            return True
        else:
            self.conformers[my_match].add_energy(result)
            return False
