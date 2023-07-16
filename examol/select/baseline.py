"""Useful baseline strategies"""
from random import sample
from typing import Iterator

import numpy as np

from .base import Selector, RankingSelector


class RandomSelector(Selector):
    """Select which computations to perform at random"""

    def __init__(self, to_select: int):
        self._options = list()
        super().__init__(to_select=to_select)

    def _add_possibilities(self, keys: list, samples: np.ndarray, **kwargs):
        self._options.extend(zip(keys, samples.mean(axis=1)))

    def _dispense(self) -> Iterator[tuple[object, float]]:
        yield from sample(self._options, min(self.to_select, len(self._options)))

    def start_gathering(self):
        super().start_gathering()
        self._options.clear()


class GreedySelector(RankingSelector):
    """Select computations which are rated the best without any regard to model uncertainty"""

    def _assign_score(self, samples):
        mean = np.mean(samples, axis=1)
        if not self.maximize:
            mean *= -1
        return mean
