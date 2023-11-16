"""Utility operations to perform common chemistry tasks"""
from functools import lru_cache

from rdkit import Chem


def parse_from_molecule_string(mol_string: str) -> Chem.Mol:
    """Parse an RDKit molecule from either SMILES or InChI

    Args:
        mol_string: String representing a molecule
    Returns:
        RDKit molecule object
    """

    if mol_string.startswith('InChI='):
        mol = Chem.MolFromInchi(mol_string, sanitize=False)
    else:
        mol = Chem.MolFromSmiles(mol_string, sanitize=False)
    if mol is None:
        raise ValueError(f'Failed to parse: "{mol_string}"')

    # Sanitize the molecule, but don't check for bad valences
    Chem.SanitizeMol(
        mol, sanitizeOps=(Chem.SANITIZE_ALL - Chem.SANITIZE_PROPERTIES)
    )

    return mol


@lru_cache()
def get_inchi_key_from_molecule_string(mol_string: str) -> str:
    """Get an InChI key from either SMILES or InChI

    Args:
        mol_string: String representing a molecule
    Returns:
        InChI key
    """
    mol = parse_from_molecule_string(mol_string)
    return Chem.MolToInchiKey(mol)


def get_baseline_charge(mol_string: str) -> int:
    """Determine the charge on a molecule from its SMILES string

    Examples:
        H<sub>2</sub>O has a baseline charge of 0
        NH<sub>4</sub>+ has a baseline charge of +1

    Args:
        mol_string: SMILES string of the molecule
    Returns:
        Charge on the molecule
    """

    mol = parse_from_molecule_string(mol_string)
    return Chem.GetFormalCharge(mol)
