"""Useful baseline strategies"""
import heapq
from random import sample
from typing import Iterator
from itertools import chain

import numpy as np

from .base import Selector


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


class GreedySelector(Selector):
    """Select computations which are rated the best without any regard to model uncertainty"""

    def __init__(self, to_select: int, maximize: bool = True):
        """
        Args:
            to_select: How many computations to select per batch
            maximize: Whether to select entries with the highest score
        """
        self._options: list[tuple[object, float]] = []
        self.maximize = maximize
        super().__init__(to_select)

    def _add_possibilities(self, keys: list, samples: np.ndarray, **kwargs):
        mean = np.mean(samples, axis=1)
        nbest = heapq.nlargest if self.maximize else heapq.nsmallest
        self._options = nbest(self.to_select, chain(self._options, zip(keys, mean)), key=lambda x: x[1])

    def _dispense(self) -> Iterator[tuple[object, float]]:
        yield from self._options

    def start_gathering(self):
        super().start_gathering()
        self._options.clear()
