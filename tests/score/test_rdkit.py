"""Tests for the RDKit scorer"""
import numpy as np
from pytest import fixture, raises
from sklearn.pipeline import Pipeline

from examol.score.rdkit import make_knn_model, RDKitScorer
from examol.store.models import MoleculeRecord


@fixture()
def pipeline() -> Pipeline:
    return make_knn_model(n_neighbors=1)


@fixture()
def scorer(recipe) -> RDKitScorer:
    return RDKitScorer(recipe)


def test_process_failure(scorer):
    record = MoleculeRecord.from_identifier('O')

    # Missing record and property
    with raises(ValueError) as err:
        scorer.transform_outputs([record])
    assert str(err.value).startswith('Record for')

    record.properties[scorer.recipe] = {}
    with raises(ValueError) as err:
        scorer.transform_outputs([record])
    assert str(err.value).startswith('Record for')


def test_transform(training_set, scorer):
    assert scorer.transform_inputs(training_set) == ['C', 'CC', 'CCC']
    assert np.isclose(scorer.transform_outputs(training_set), [1, 2, 3]).all()


def test_functions(training_set, scorer, pipeline):
    model_msg = scorer.prepare_message(pipeline)
    assert isinstance(model_msg, Pipeline)

    # Test training
    inputs = scorer.transform_inputs(training_set)
    outputs = scorer.transform_outputs(training_set)
    update_msg = scorer.retrain(model_msg, inputs, outputs)
    pipeline, scorer.update(pipeline, update_msg)

    # Test scoring
    model_msg = scorer.prepare_message(pipeline)
    scores = scorer.score(model_msg, inputs)
    assert np.isclose(scores, outputs).all()
