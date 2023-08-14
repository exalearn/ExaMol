"""Algorithms which select molecules based on k-means clustering"""
import numpy as np
from sklearn.cluster import KMeans
from .base import Starter
from examol.score.rdkit.descriptors import compute_morgan_fingerprints
from scipy.spatial.distance import cdist


class KMeansStarter(Starter):
    """Select structurally distinct molecules by picking molecules at the centers of clusters"""

    def _select(self, to_select: list[str], count: int) -> list[str]:
        # Compute Morgan fingerprints along with their indices
        fingerprints = [compute_morgan_fingerprints(smiles) for smiles in to_select]

        # Run KMeans clustering on the fingerprints
        kmeans = KMeans(n_clusters=count).fit(fingerprints)

        # Get cluster centers
        cluster_centers = kmeans.cluster_centers_

        # Get the molecules closest to the cluster centers
        # Using cdist to compute the distance between each pair of fingerprints and cluster centers
        distances = cdist(fingerprints, cluster_centers, 'euclidean')

        closest_molecules = []
        # For each cluster center, find the molecule with the smallest distance to it
        for i in range(count):
            closest_index = np.argmin(distances[:, i])
            closest_molecules.append(to_select[closest_index])

        return closest_molecules
