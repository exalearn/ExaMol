"""Implementations of classes which identify which computations should be performed next"""
from typing import Iterator

import numpy as np


class Selector:
    """Base class for selection algorithms

    **Using a Selector**

    Selectors function in two phases: a gathering and a dispensing.

    The gathering phase starts by calling :meth:`start_gathering` before adding new options for computations
    with the :meth:`add_possibilities` option. ``add_possibilities`` takes a list of keys describing the computations
    and a distribution of possible scores (e.g., predictions from different models in an ensemble) for each computation.

    The dispensing phase starts after :meth:`start_dispensing` is called, which makes it then possible to pull
    a list of prioritized computations from :meth:`dispense`. ``dispense`` generates a selected computation from
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
            keys: Labels by which to identify the compositions being selected between
            samples: A distribution of scores for each record. For example,
                these could be predictions of its properties from a
        """
        assert self.gathering, 'Not in gathering phase. Call `start_gathering` first'
        assert len(keys) == len(samples), 'The list of keys and samples should be the same length'
        self._add_possibilities(keys, samples, **kwargs)

    def _add_possibilities(self, keys: list, samples: np.ndarray, **kwargs):
        raise NotImplementedError()

    def start_dispensing(self):
        """Prepare to generate batches of new computations"""
        self.gathering = False

    def dispense(self) -> Iterator[tuple[object, float]]:
        """Dispense selected computations from highest- to least-rated.

        Yields:
            A pair of "selected computation" (as identified by the keys provided originally)
            and a score.
        """
        assert not self.gathering, 'Not in dispensing phase. Call `start_dispensing` first'
        yield from self._dispense()

    def _dispense(self) -> Iterator[tuple[object, float]]:
        raise NotImplementedError()
