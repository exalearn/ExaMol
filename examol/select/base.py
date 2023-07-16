"""Implementations of classes which identify which computations should be performed next"""
import heapq
from itertools import chain
from typing import Iterator

import numpy as np

from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


class Selector:
    """Base class for selection algorithms

    **Using a Selector**

    Selectors function in two phases: gathering and dispensing.

    The gathering phase starts by calling :meth:`start_gathering` to clear any data from previous runs
    before adding new options for computations with :meth:`add_possibilities`.
    ``add_possibilities`` takes a list of keys describing the computations
    and a distribution of possible scores (e.g., predictions from different models in an ensemble) for each computation.

    The dispensing phase starts by calling :meth:`dispense`. ``dispense`` generates a selected computation from
    the list of keys acquired during gathering phase paired with a score. Selections are generated from highest
    to lowest priority.
    """

    def __init__(self, to_select: int):
        """

        Args:
            to_select: Target number of computations to select
        """
        self.to_select = to_select
        self.gathering = True
        self.start_gathering()

    def start_gathering(self):
        """Prepare to gather new batches potential computations"""
        self.gathering = True

    def add_possibilities(self, keys: list, samples: np.ndarray, **kwargs):
        """Add potential options to be selected

        Args:
            keys: Labels by which to identify the records being evaluated
            samples: A distribution of scores for each record.
                Expects a two-dimensional array where each row is a different record,
                and each column is a different model.
        """
        assert self.gathering, 'Not in gathering phase. Call `start_gathering` first'
        assert len(keys) == len(samples), 'The list of keys and samples should be the same length'
        self._add_possibilities(keys, samples, **kwargs)

    def _add_possibilities(self, keys: list, samples: np.ndarray, **kwargs):
        raise NotImplementedError()

    def update(self, database: dict[str, MoleculeRecord], recipe: PropertyRecipe):
        """Update the selector given the current database

        Args:
            database: Known molecules
            recipe: Recipe being optimized
        """
        pass

    def dispense(self) -> Iterator[tuple[object, float]]:
        """Dispense selected computations from highest- to least-rated.

        Yields:
            A pair of "selected computation" (as identified by the keys provided originally)
            and a score.
        """
        self.gathering = False
        yield from self._dispense()

    def _dispense(self) -> Iterator[tuple[object, float]]:
        raise NotImplementedError()


class RankingSelector(Selector):
    """Base class where we assign an independent score to each possibility.

    Implementations must return high scores for desired entries
    regardless of whether the the selector is set to minimize.

    Args:
        to_select: How many computations to select per batch
        maximize: Whether to select entries with high or low values of the samples
    """
    def __init__(self, to_select: int, maximize: bool = True):
        self._options: list[tuple[object, float]] = []
        self.maximize = maximize
        super().__init__(to_select)

    def _add_possibilities(self, keys: list, samples: np.ndarray, **kwargs):
        score = self._assign_score(samples)
        self._options = heapq.nlargest(self.to_select, chain(self._options, zip(keys, score)), key=lambda x: x[1])

    def _dispense(self) -> Iterator[tuple[object, float]]:
        yield from self._options

    def start_gathering(self):
        super().start_gathering()
        self._options.clear()

    def _assign_score(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
