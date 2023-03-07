"""Scorers that rely on RDKit and sklearn"""
import pickle as pkl

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, DataStructs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

from examol.score.base import Scorer
from examol.store.models import MoleculeRecord


class RDKitScorer(Scorer):
    """Score molecules based on a model defined using RDKit and Scikit-Learn"""

    def __init__(
            self,
            output_property: str,
            output_level: str,
            pipeline: Pipeline,
    ):
        """

        Args:
            output_property: Property to be predicted
            output_level: Level at which to predict it
            pipeline: sklearn regression model which takes a `RDKit.mol` as inputs
        """
        self.output_property = output_property
        self.output_level = output_level
        self.model = pipeline

    def transform_inputs(self, record_batch: list[MoleculeRecord]) -> list:
        return [x.identifier.smiles for x in record_batch]

    def transform_outputs(self, records: list[MoleculeRecord]) -> np.ndarray:
        return np.array([x.properties[self.output_property][self.output_level] for x in records])

    def get_model_state(self) -> object:
        return pkl.dumps(self.model)

    @staticmethod
    def score(model_msg: object, inputs: list, **kwargs) -> np.ndarray:
        model: Pipeline = pkl.loads(model_msg)
        return model.predict(inputs)

    @staticmethod
    def retrain(model_msg: object, inputs: list, outputs: list, **kwargs) -> object:
        model: Pipeline = pkl.loads(model_msg)
        model.fit(inputs, outputs)
        return pkl.dumps(model)

    def update(self, update_msg: object):
        self.model = pkl.loads(update_msg)


def make_knn_model(n_neighbors: int = 2, length: int = 256, radius: int = 4, **kwargs) -> Pipeline:
    """Make a KNN model based on Morgan Fingerprints

    Args:
        n_neighbors: Number of neighbors to consider in KNN
        length: Length of the fingerprint vector
        radius: Radius of the fingerprint computation
    """

    return Pipeline([
        ('parse', MoleculeParserTransformer()),
        ('fingerprint', MorganFingerprintTransformer(length, radius)),
        ('knn', KNeighborsRegressor(n_neighbors=n_neighbors, metric='jaccard', n_jobs=1))
    ])


class MoleculeParserTransformer(BaseEstimator, TransformerMixin):
    """Parses a molecule from SMILES string to RDKit.Mol"""

    def fit(self, X, y=None):
        return self  # Do nothing

    def transform(self, X, y=None):
        """Compute the fingerprints

        Args:
            X: List of SMILES strings
        Returns:
            Array of fingerprints
        """

        output = []
        for smiles in X:
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None, f'Parsing failed for {smiles}'
            output.append(mol)
        return output


class MorganFingerprintTransformer(BaseEstimator, TransformerMixin):
    """Class that converts RDKit Mols to fingerprint vectors"""

    def __init__(self, length: int = 256, radius: int = 4):
        """

        Args:
            length: Length of the bit vector
            radius: Graph radius to consider
        """
        self.length = length
        self.radius = radius

    def fit(self, X, y=None):
        return self  # Do nothing

    def compute_morgan_fingerprints(self, mol: Chem.Mol) -> np.ndarray:
        """Compute the Morgan fingerprints for a molecule

        Args:
            mol: Molecule to be fingerprinted
        Returns:
            Fingerprint vector
        """
        # Compute the fingerprint
        fingerprint = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol, self.radius, self.length
        )
        arr = np.zeros((1,), dtype=bool)
        DataStructs.ConvertToNumpyArray(fingerprint, arr)
        return arr

    def transform(self, X, y=None):
        """Compute the fingerprints

        Args:
            X: List of SMILES strings
        Returns:
            Array of fingerprints
        """

        fing = [self.compute_morgan_fingerprints(m) for m in X]
        return np.vstack(fing)
