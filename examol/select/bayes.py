"""Acquisition functions derived from Bayesian optimization"""

import numpy as np
from modAL.acquisition import EI

from examol.select.base import RankingSelector
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

    def update(self, database: dict[str, MoleculeRecord], recipe: PropertyRecipe):
        values = [
            recipe.lookup(x) for x in database.values() if recipe.lookup(x) is not None
        ]
        self.best_so_far = max(values) if self.maximize else min(values)

    def _assign_score(self, samples: np.ndarray) -> np.ndarray:
        # Compute the mean and standard deviation for each entry
        mean = samples.mean(axis=1)
        std = samples.std(axis=1)

        # If we are minimizing, invert the values
        max_y = self.best_so_far
        if not self.maximize:
            mean *= -1
            max_y = max_y * -1

        return EI(mean, std, max_val=max_y, tradeoff=self.epsilon)
