"""Utilities used for multi-fidelity learning"""
from typing import Sequence

import numpy as np

from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


def collect_outputs(records: list[MoleculeRecord], recipes: Sequence[PropertyRecipe]) -> np.ndarray:
    """Collect the outputs for several recipe for each molecule

    Args:
        records: Molecule records to be summarized
        recipes: List of recipes to include
    Returns:
        Matrix where each row is a different molecule, and each column is a different recipe.
        Values where the recipe is unknown are NaN.
    """
    return np.array([
        [record.properties.get(recipe.name, {}).get(recipe.level, np.nan) for recipe in recipes]
        for record in records
    ])


def compute_deltas(known_outputs: np.ndarray) -> np.ndarray:
    """Compute the deltas between individual levels given a numpy array

    Args:
        known_outputs: ``num_entries x num_levels`` Values for a property at each level of fidelity.
    Returns:
        An array where the first column is the value at the lowest level,
        and the other columns are differences between its value and that of the previous level of fidelity
    """
    deltas = np.array(known_outputs)
    deltas[:, 1:] = np.diff(deltas)
    return deltas
