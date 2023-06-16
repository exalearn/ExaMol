"""Python wrapper which computes large numbers of features from RDKit

Adapted from `COBOL <https://github.com/MolecularMaterials/COBOL/blob/7276cca428ac1acdf44ac0d580e9e70e64ac642c/cobol/src/utils/chem_utils.py#L57>`_"""
from typing import Callable

import numpy as np
from rdkit import Chem
import rdkit.Chem.Fragments as Fragments
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.Lipinski as Lipinski
import rdkit.Chem.rdMolDescriptors as MolDescriptors

# List of descriptors, populated once
_descriptor_list: list[tuple[str, Callable]] = []

_descriptor_list.extend((name, getattr(Fragments, name)) for name in dir(Fragments) if "fr" in name)
_descriptor_list.extend((name, getattr(Lipinski, name)) for name in dir(Lipinski) if ("Count" in name and "Smarts" not in name))
_descriptor_list.extend((name, getattr(MolDescriptors, name)) for name in dir(MolDescriptors)
                        if (("CalcNum" in name or "CalcExact" in name or "CalcTPSA" in name or "CalcChi" in name or "Kappa" in name or "Labute" in name)
                            and "Stereo" not in name and "_" not in name and "ChiN" not in name))
_descriptor_list.extend((name, getattr(Crippen, name)) for name in dir(Crippen)
                        if (("MolLogP" in name or "MolMR" in name) and "_" not in name))

# Remove ring-related descriptors
_descriptor_list = sorted(
    (name, func) for name, func in _descriptor_list
    if name not in ['CalcNumAromaticHeterocycles', 'CalcNumSaturatedHeterocycles', 'CalcNumSaturatedRings',
                    'CalcNumAromaticCarbocycles', 'CalcNumRings', 'CalcNumHeavyAtoms', 'CalcNumRotatableBonds',
                    'CalcNumAliphaticRings', 'CalcNumHeteroatoms', 'CalcNumAromaticRings', 'CalcNumAliphaticHeterocycles']
)


def compute_doan_2020_fingerprints(smiles: str) -> np.ndarray:
    """Compute fingerprints from `Doan et al. <https://pubs.acs.org/doi/10.1021/acs.chemmater.0c00768>`_

    Args:
        smiles: SMILES string to be evaluated
    Returns:
        Molecular descriptors
    """
    output = np.zeros((len(_descriptor_list),))
    mol = Chem.MolFromSmiles(smiles)
    for i, (_, func) in enumerate(_descriptor_list):
        try:
            output[i] = func(mol)
        except Exception:
            pass
    return output
