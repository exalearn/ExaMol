"""Scorers that rely on RDKit and sklearn"""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from examol.score.base import Scorer
from examol.score.rdkit.descriptors import compute_morgan_fingerprints
from examol.store.models import MoleculeRecord


class RDKitScorer(Scorer):
    """Score molecules based on a model defined using RDKit and Scikit-Learn

    Models must take a SMILES string as input.
    Use the :class:`~.MoleculeParserTransformer` to transform the SMILES into an RDKit Mol object if needed.
    """

    def transform_inputs(self, record_batch: list[MoleculeRecord]) -> list:
        return [x.identifier.smiles for x in record_batch]

    def prepare_message(self, model: Pipeline, training: bool = True) -> Pipeline:
        return model

    def score(self, model_msg: Pipeline, inputs: list, **kwargs) -> np.ndarray:
        return model_msg.predict(inputs)

    def retrain(self, model_msg: Pipeline, inputs: list, outputs: np.ndarray, **kwargs) -> object:
        model_msg.fit(inputs, outputs)
        return model_msg

    def update(self, model: object, update_msg: object) -> object:
        return update_msg


def make_knn_model(n_neighbors: int = 2, length: int = 256, radius: int = 4, n_jobs: int | None = None) -> Pipeline:
    """Make a KNN model based on Morgan Fingerprints

    Args:
        n_neighbors: Number of neighbors to consider in KNN
        length: Length of the fingerprint vector
        radius: Radius of the fingerprint computation
        n_jobs: Number of processes to use for computing fingerprints and nearest neighbors
    """

    func = partial(compute_morgan_fingerprints, length=length, radius=radius)
    return Pipeline([
        ('fingerprint', FingerprintTransformer(func, n_jobs=n_jobs)),
        ('knn', KNeighborsRegressor(n_neighbors=n_neighbors, metric='jaccard', n_jobs=n_jobs))
    ])


class FingerprintTransformer(BaseEstimator, TransformerMixin):
    """Class that converts RDKit Mols to fingerprint vectors

    Args:
        function: Function which takes a screen and generates a
    """

    def __init__(self, function: Callable[[str], np.ndarray], n_jobs: int | None = None):
        self.func = function
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        return self  # Do nothing

    def transform(self, X, y=None):
        """Compute the fingerprints

        Args:
            X: List of SMILES strings
        Returns:
            Array of fingerprints
        """

        with ProcessPoolExecutor(max_workers=self.n_jobs) as pool:
            fing = pool.map(self.func, X)
        return np.vstack(fing)
