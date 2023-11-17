"""Scorers that rely on RDKit and sklearn"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Union

import numpy as np
from sklearn import clone
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process import kernels
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from examol.score.base import MultiFidelityScorer
from examol.score.utils.multifi import compute_deltas
from examol.score.rdkit.descriptors import compute_morgan_fingerprints, compute_doan_2020_fingerprints
from examol.store.models import MoleculeRecord

ModelType = Pipeline | list[Pipeline]
"""Model is a single for training and a list of models after training"""
InputType = list[str]
"""Model inputs are the SMILES string of the molecule"""


@dataclass
class RDKitScorer(MultiFidelityScorer):
    """Score molecules based on a model defined using RDKit and Scikit-Learn

    Models must take a SMILES string as input.
    Use the :class:`~.FingerprintTransformer` to transform the SMILES into an RDKit Mol object if needed.

    **Multi Fidelity Learning**

    We implement multi-fidelity learning by training separate models for each level of fidelity.

    The model for the lowest level of fidelity is trained to predict the value of the property
    and each subsequent model predicts the delta between it and the previous step.

    On inference, we use the known values for either the lowest level of fidelity or
    the deltas in place of the predictions from the machine learning models.
    """

    def transform_inputs(self, record_batch: list[MoleculeRecord]) -> InputType:
        return [x.identifier.smiles for x in record_batch]

    def prepare_message(self, model: ModelType, training: bool = True) -> ModelType:
        if training:
            # Only send a single model for training
            return model[0] if isinstance(model, list) else model
        else:
            # Send the whole list for inference
            return model

    def score(self, model_msg: ModelType, inputs: InputType, lower_fidelities: np.ndarray | None = None, **kwargs) -> np.ndarray:
        if not isinstance(model_msg, list):
            # Single objective
            return model_msg.predict(inputs)
        else:
            # Get the known deltas then append a NaN to the end (we don't know the last delta)
            if lower_fidelities is None:
                deltas = np.empty((len(inputs), len(model_msg))) * np.nan
            else:
                known_deltas = compute_deltas(lower_fidelities)
                deltas = np.concatenate((known_deltas, np.empty_like(known_deltas[:, :1]) * np.nan), axis=1)

            # Run the model at each level
            for my_level, my_model in enumerate(model_msg):
                my_preds = my_model.predict(inputs)
                is_unknown = np.isnan(deltas[:, my_level])
                deltas[is_unknown, my_level] = my_preds[is_unknown]

            # Sum up the deltas
            return np.sum(deltas, axis=1)

    def retrain(self, model_msg: Pipeline, inputs: InputType, outputs: np.ndarray,
                bootstrap: bool = True,
                lower_fidelities: np.ndarray | None = None) -> ModelType:
        if bootstrap:
            samples = np.random.random_integers(0, len(inputs) - 1, size=(len(inputs),))
            inputs = [inputs[i] for i in samples]
            outputs = outputs[samples]
            if lower_fidelities is not None:
                lower_fidelities = lower_fidelities[samples, :]

        if lower_fidelities is None:
            # For single level, train a single model
            model_msg.fit(inputs, outputs)
            return model_msg
        else:
            # Compute the delta and then train a different model for each delta
            outputs = np.concatenate([lower_fidelities, outputs[:, None]], axis=1)  # Append target level to end
            deltas = compute_deltas(outputs)

            models = []
            for y in deltas.T:
                # Remove the missing values
                mask = np.isfinite(y)
                my_smiles = [i for m, i in zip(mask, inputs) if m]
                y = y[mask]

                # Fit a fresh copy of the model
                my_model: Pipeline = clone(model_msg)
                my_model.fit(my_smiles, y)
                models.append(my_model)
            return models

    def update(self, model: ModelType, update_msg: ModelType) -> ModelType:
        return update_msg


def make_knn_model(n_neighbors: int = 2, length: int = 256, radius: int = 4, n_jobs: int | None = None) -> Pipeline:
    """Make a KNN model based on Morgan Fingerprints

    Args:
        n_neighbors: Number of neighbors to consider in KNN
        length: Length of the fingerprint vector
        radius: Radius of the fingerprint computation
        n_jobs: Number of processes to use for computing fingerprints and nearest neighbors
    Returns:
        Pipeline which predicts outputs based on SMILES string
    """

    func = partial(compute_morgan_fingerprints, length=length, radius=radius)
    return Pipeline([
        ('fingerprint', FingerprintTransformer(func, n_jobs=n_jobs)),
        ('knn', KNeighborsRegressor(n_neighbors=n_neighbors, metric='jaccard', n_jobs=n_jobs))
    ])


def make_gpr_model(num_pcs: int | None = None, max_pcs: int = 10, k: int = 3) -> Union[GridSearchCV, Pipeline]:
    """Make a Gaussian process regression model using the features of
    `Doan et al. <https://pubs.acs.org/doi/10.1021/acs.chemmater.0c00768>`_
    and feature selection based on PCA

    Args:
        num_pcs: Number of principal components to use. Set to ``None`` to fit with k-fold cross-validation
        max_pcs: Maximum number of principal components to use
        k: Number of folds to use when fitting the component count

    Returns:
        Pipeline which predicts outputs based on SMILES string
    """

    # Set up the kernel
    kernel = kernels.ConstantKernel(1.0, (1e-3, 1e3)) * kernels.Matern() + kernels.WhiteKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=40, normalize_y=False)

    # Make the pipeline
    pipeline = Pipeline([
        ('fingerprints', FingerprintTransformer(compute_doan_2020_fingerprints)),
        ('pca', PCA(n_components=num_pcs)),
        ('gpr', gp)
    ])

    if num_pcs is None:
        pipeline = GridSearchCV(
            pipeline,
            param_grid={'pca__n_components': range(1, max_pcs + 1)},
            cv=k,
            n_jobs=1  # Parallelism is a level below
        )
    return pipeline


class FingerprintTransformer(BaseEstimator, TransformerMixin):
    """Pipeline step which converts SMILES strings to fingerprint vectors

    Args:
        function: Function which takes a SMILES string and generates a vector of fingerprints
        n_jobs: Number of fingerprinting tasks to run in parallel
    """

    def __init__(self, function: Callable[[str], np.ndarray], n_jobs: int | None = None):
        self.function = function
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
            fing = list(pool.map(self.function, X))
        return np.vstack(fing)
