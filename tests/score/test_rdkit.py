"""Tests for the RDKit scorer"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from pytest import fixture, mark
from sklearn.pipeline import Pipeline

from examol.score.rdkit.descriptors import compute_doan_2020_fingerprints
from examol.score.rdkit import make_knn_model, RDKitScorer, make_gpr_model


@fixture()
def pipeline() -> Pipeline:
    return make_knn_model(n_neighbors=1)


@fixture()
def scorer() -> RDKitScorer:
    return RDKitScorer()


def test_transform(training_set, scorer, recipe):
    assert scorer.transform_inputs(training_set)[0][0] == 'C'
    assert np.isclose(scorer.transform_outputs(training_set, recipe), [1, 2, 3]).all()


@mark.parametrize('bootstrap', [True, False])
def test_functions(training_set, scorer, pipeline, recipe, bootstrap):
    model_msg = scorer.prepare_message(pipeline)
    assert isinstance(model_msg, Pipeline)

    # Test training
    inputs = scorer.transform_inputs(training_set)
    outputs = scorer.transform_outputs(training_set, recipe)
    update_msg = scorer.retrain(model_msg, inputs, outputs, bootstrap=bootstrap)
    pipeline, scorer.update(pipeline, update_msg)

    # Test scoring
    model_msg = scorer.prepare_message(pipeline)
    scores = scorer.score(model_msg, inputs)
    if not bootstrap:
        assert np.isclose(scores, outputs).all()  # KNN should fit the dataset perfectly


def test_doan_descriptors():
    x = compute_doan_2020_fingerprints('C')
    with ProcessPoolExecutor(mp_context=get_context('spawn')) as p:
        y = p.submit(compute_doan_2020_fingerprints, 'C').result()
    assert np.isclose(x, y).all()


def test_gpr(training_set, scorer, recipe):
    pipeline = make_gpr_model()

    # Test training
    model_msg = scorer.prepare_message(pipeline)
    inputs = scorer.transform_inputs(training_set)
    outputs = scorer.transform_outputs(training_set, recipe)
    update_msg = scorer.retrain(model_msg, inputs, outputs, bootstrap=False)
    pipeline, scorer.update(pipeline, update_msg)

    assert pipeline.best_estimator_.steps[1][1].n_components < 10


@mark.parametrize('bootstrap', [True, False])
def test_multifi(training_set, multifi_recipes, scorer, pipeline, bootstrap):
    # Test conversion to multi-fidelity
    inputs = scorer.transform_inputs(training_set, multifi_recipes)
    assert inputs[0][0] == training_set[0].identifier.smiles
    assert inputs[0][1] is not None

    # Test training
    model_msg = scorer.prepare_message(pipeline, training=True)
    outputs = scorer.transform_outputs(training_set, multifi_recipes)
    update_msg = scorer.retrain(model_msg, inputs, outputs, bootstrap=bootstrap)
    assert len(update_msg) == len(multifi_recipes)

    # Test updating
    pipeline = scorer.update(pipeline, update_msg)
    assert len(pipeline) == len(multifi_recipes)

    # Test inference
    model_msg = scorer.prepare_message(pipeline, training=False)
    outputs = scorer.score(model_msg, inputs)
    assert outputs.shape == (len(training_set),)
    values = [v[-1] for _, v in inputs]
    assert np.isclose(outputs, values).all()  # Should give exact result, since all values are known
