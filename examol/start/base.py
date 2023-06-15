"""Base class and utilities"""
from typing import Iterable

from more_itertools import take


class Starter:
    """Template for classes which select an initial set of calculations to run.

    Args:
        threshold: Run the starter if the database is smaller than this number
        min_to_select: Minimum number of molecules to select
        max_to_consider: Only select from the up to the first ``max_to_consider`` molecules
            in the search space, if set.
    """

    def __init__(self, threshold: int, min_to_select: int, max_to_consider: int | None = None):
        self.threshold = threshold
        self.min_to_select = min_to_select
        self.max_to_consider = max_to_consider

    def select(self, to_select: Iterable[str], count: int) -> list[str]:
        """Select a subset of molecules to run

        Args:
            to_select: Iterator of SMILES strings
            count: Minimum number of select
        Returns:
            List of at least :attr:`min_to_select` chosen SMILES strings
        """
        # Determine how many to pull
        count = max(self.min_to_select, count)

        # Get the pool to draw from
        if self.max_to_consider is not None:
            pool = take(self.max_to_consider, to_select)
        else:
            pool = list(to_select)

        return self._select(pool, count)

    def _select(self, to_select: list[str], count: int) -> list[str]:
        """Perform the selection

        Args:
            to_select: List from which to draw candidates
            count: Number to draw
        Returns:
            Selected subset
        """
        raise NotImplementedError()
