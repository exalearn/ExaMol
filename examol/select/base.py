"""Implementations of classes which identify which computations should be performed next"""
from typing import Iterator

import numpy as np


class Selector:
    """Base class for selection algorithms

    **Using a Selector**

    Selectors function in two phases: a gathering and a dispensing.

    The gathering phase starts by calling :meth:`start_gathering` before adding new options for computations
    with the :meth:`add_possibilities` option.

    The dispensing phase starts after :meth:`start_dispensing` is called, which makes it
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
            samples: Samples of the potential property values
        """
        assert self.gathering, 'Not in gathering phase. Call `start_gathering` first'
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
