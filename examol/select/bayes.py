"""Acquisition functions derived from Bayesian optimization"""
from typing import Sequence

import numpy as np
from modAL.acquisition import EI

from examol.select.base import RankingSelector, _extract_observations
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


class ExpectedImprovement(RankingSelector):
    """Rank entries according to their expected improvement

     Args:
        to_select: How many computations to select per batch
        maximize: Whether to select entries with the highest score
        epsilon: Parameter which controls degree of exploration
    """

    def __init__(self, to_select: int, maximize: bool, epsilon: float = 0):
        super().__init__(to_select, maximize)
        self.epsilon = epsilon
        self.best_so_far = 0

    def update(self, database: dict[str, MoleculeRecord], recipes: Sequence[PropertyRecipe]):
        values = _extract_observations(database, recipes)
        self.best_so_far = max(values) if self.maximize else -min(values)

    def _assign_score(self, samples: np.ndarray) -> np.ndarray:
        # Compute the mean and standard deviation for each entry
        mean = samples.mean(axis=(0, 2))
        std = samples.std(axis=(0, 2))

        # If we are minimizing, invert the values
        max_y = self.best_so_far
        return EI(mean, std, max_val=max_y, tradeoff=self.epsilon)
