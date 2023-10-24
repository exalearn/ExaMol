"""Base class and utilities"""
from typing import Collection

from more_itertools import take


class Starter:
    """Template for classes which select an initial set of calculations to run.

    Args:
        max_to_consider: Only select from the up to the first ``max_to_consider`` molecules
            in the search space, if set.
    """

    def __init__(self, max_to_consider: int | None = None):
        self.max_to_consider = max_to_consider

    def select(self, to_select: Collection[str], count: int) -> list[str]:
        """Select a subset of molecules to run

        Args:
            to_select: Collection of SMILES strings
            count: Number of computations to select
        Returns:
            List of at least :attr:`min_to_select` chosen SMILES strings
        """
        # Make sure there are enough molecules
        if count > len(to_select):
            raise ValueError(f"Cannot select {count} molecules from {len(to_select)} molecules.")

        # Get the pool to draw from (TODO: draw randomly rather than the first)
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
