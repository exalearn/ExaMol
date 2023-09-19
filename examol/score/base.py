"""Base classes for scoring functions"""
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


def collect_outputs(records: list[MoleculeRecord], recipes: list[PropertyRecipe]) -> np.ndarray:
    """Collect the outputs for several recipe for each molecule

    Args:
        records: Molecule records to be summarized
        recipes: List of recipes to include
    Returns:
        Matrix where each row is a different molecule, and each column is a different recipe
    """
    return np.array([
        [record.properties.get(recipe.name, {}).get(recipe.level, np.nan) for recipe in recipes]
        for record in records
    ])


@dataclass
class Scorer:
    """Base class for algorithms which quickly assign a score to a molecule, typically using a machine learning model

    **Using a Scorer**

    Scoring a molecule requires transforming the molecule into a form compatible with a machine learning algorithm,
    then executing inference using the machine learning algorithm.
    We separate these two steps so that the former can run on local resources
    and the latter can run on larger remote resource.
    Running the scorer will then look something like

    .. code-block:: python

        scorer = Scorer()
        recipe = PropertyRecipe()  # Recipe that we are trying to predict
        model = ...   # The model that we'll be sending to workers
        inputs = model.transform_inputs(records)  # Readies records to run inference
        model_msg = model.prepare_message(model)  # Readies model to be sent to a remote worker
        scorer.score(model_msg, inputs)  # Can be run remotely

    Note how the ``Scorer`` class does not hold on to the model as state.
    The Scorer is just the tool which holds code needed train and run the model.

    Training operations are broken into separate operations for similar reasons.
    We separate the training operation from pre-processing inputs and outputs,
    and updating a local copy of the model given the results of training.

    .. code-block: python

        outputs = scorer.transform_outputs(records, recipe)  # Prepares label for a specific recipe
        update_msg = scorer.retrain(model_msg, inputs, outputs)  # Run remotely
        model = scorer.update(model, update_msg)

    **Multi-fidelity scoring**

    Multi-fidelity learning methods employ lower-fidelity estimates of a target value to improve the prediction of that value.
    ExaMol supports multi-fidelity through the ability to provide more than one recipe as inputs to
    :meth:`transform_inputs` and :meth:`transform_outputs`.

    Implementations of Scorers must be designed to support multi-fidelity learning.
    """

    _supports_multi_fidelity: bool = False
    """Whether the class supports multi-fidelity optimization"""

    def transform_inputs(self, record_batch: list[MoleculeRecord], recipes: Sequence[PropertyRecipe] | None = None) -> list:
        """Form inputs for the model based on the data in a molecule record

        Args:
            record_batch: List of records to pre-process
            recipes: List of recipes ordered from lowest to highest fidelity.
                Only used in multi-fidelity scoring algorithms
        Returns:
            List of inputs ready for :meth:`score` or :meth:`retrain`
        """
        raise NotImplementedError()

    # TODO (wardlt): I'm not super-happy with multi-fidelity being inferred from input types. What if we want multi-objective learning
    def transform_outputs(self, records: list[MoleculeRecord], recipe: PropertyRecipe | Sequence[PropertyRecipe]) -> np.ndarray:
        """Gather the target outputs of the model

        Args:
            records: List of records from which to extract outputs
            recipe: Target recipe for the scorer for single-fidelity learning
                or a list of recipes ordered from lowest to highest fidelity
                for multi-objective learning.
        Returns:
            Outputs ready for model training
        """
        # Determine if we are doing single or multi-fidelity learning
        is_single = False
        if isinstance(recipe, PropertyRecipe):
            is_single = True
            recipes = [recipe]
        else:
            if not self._supports_multi_fidelity:  # program: no-coverage
                raise ValueError(f'{self.__class__.__name__} does not support multi-fidelity training')
            recipes = recipe

        # Gather the outputs
        output = collect_outputs(records, recipes)
        if is_single:
            return output[:, 0]
        return output

    def prepare_message(self, model: object, training: bool = False) -> object:
        """Get the model state as a serializable object

        Args:
            model: Model to be sent to `score` or `retrain` function
            training: Whether to prepare the message for training or inference
        Returns:
            Get the model state as an object which can be serialized then transmitted to a remote worker
        """
        raise NotImplementedError()

    def score(self, model_msg: object, inputs: list, **kwargs) -> np.ndarray:
        """Assign a score to molecules

        Args:
            model_msg: Model in a transmittable format, may need to be deserialized
            inputs: Batch of inputs ready for the model, as generated by :meth:`transform_inputs`
        Returns:
            The scores to a set of records
        """
        raise NotImplementedError()

    def retrain(self, model_msg: object, inputs: list, outputs: list, **kwargs) -> object:
        """Retrain the scorer based on new training records

        Args:
            model_msg: Model to be retrained
            inputs: Training set inputs, as generated by :meth:`transform_inputs`
            outputs: Training Set outputs, as generated by :meth:`transform_outputs`
        Returns:
            Message defining how to update the model
        """
        raise NotImplementedError()

    def update(self, model: object, update_msg: object) -> object:
        """Update this local copy of a model

        Args:
            model: Model to be updated
            update_msg: Update for the model
        Returns:
            Updated model
        """
        raise NotImplementedError()
