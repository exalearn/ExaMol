"""Base class and utilities"""
from typing import Iterable


class Starter:
    """Template for classes which select an initial set of calculations to run.

    Args:
        threshold: Run the starter if the database is smaller than this number
        to_select: Number of molecules to select
        max_to_consider: Only select from the up to the first ``max_to_consider`` molecules
            in the search space, if set.
    """

    def __init__(self, threshold: int, to_select: int, max_to_consider: int | None = None):
        self.threshold = threshold
        self.to_select = to_select
        self.max_to_consider = max_to_consider

    def select(self, to_select: Iterable[str]) -> list[str]:
        """Select a subset of molecules to run

        Args:
            to_select: Iterator of SMILES strings
        Returns:
            List of chosen SMILES strings
        """
        raise NotImplementedError()
