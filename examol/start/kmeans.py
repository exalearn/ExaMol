"""Algorithms which select molecules based on k-means clustering"""
from .base import Starter
from examol.score.rdkit.descriptors import compute_morgan_fingerprints


class KMeansStarter(Starter):
    """Select structurally distinct molecules by picking molecules at the centers of clusters"""

    def _select(self, to_select: list[str], count: int) -> list[str]:
        # Compute the fingerprints for each molecule
        finger = [
            compute_morgan_fingerprints(x) for x in to_select
        ]

        # Pick the molecules at the center of each cluster
        raise NotImplementedError()
