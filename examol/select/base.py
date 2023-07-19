"""Implementations of classes which identify which computations should be performed next"""
import heapq
import logging
from itertools import chain
from typing import Iterator

import numpy as np

from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe

logger = logging.getLogger(__name__)


class Selector:
    """Base class for selection algorithms

    **Using a Selector**

    Selectors function in two phases: gathering and dispensing.

    Selectors are in the gathering phase when first created.
    Add potential computations in batches with :meth:`add_possibilities`,
    which takes a list of keys describing the computations
    and a distribution of probable scores (e.g., predictions from different models in an ensemble) for each computation.
    Sample arrays are 3D and shaped ``num_recipes x num_samples x num_models``

    The dispensing phase starts by calling :meth:`dispense`. ``dispense`` generates a selected computation from
    the list of keys acquired during gathering phase paired with a score. Selections are generated from highest
    to lowest priority.

    **Creating a Selector**

    You must implement three operations:

    - :meth:`start_gathering`, which is called at the beginning of a gathering phase and
      must clear state from the previous selection round.
    - :meth:`add_possibilities` updates the state of a selection to account for a new batch of computations.
      For example, you could update an ranked list of best-scored computations.
    - :meth:`dispense` generates a list of :attr:`to_select` in ranked order from best to worst
    """

    multiobjective: bool = False
    """Whether the selector supports multi-objective optimization"""

    def __init__(self, to_select: int):
        """

        Args:
            to_select: Target number of computations to select
        """
        self.to_select: int = to_select
        """Number of computations to select"""
        self.gathering: bool = True
        """Whether the selector is waiting to accept more possibilities."""
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
        if samples.shape[0] > 1 and not self.multiobjective:
            raise ValueError(f'Provided {samples.shape[0]} objectives but the class does not support multi-objective selection')
        if samples.ndim != 3:  # pragma: no-coverage
            raise ValueError(f'Expected samples dimension of 3. Found {samples.ndim}. Array should be (recipe, sample, model)')
        if samples.shape[1] != len(keys):
            raise ValueError(f'Number of keys and number of samples differ. Keys={len(keys)}. Samples={samples.shape[1]}')
        if not self.gathering:
            logger.info('Switching selector back to gathering phase. Clearing any previous selection information')
            self.start_gathering()
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
