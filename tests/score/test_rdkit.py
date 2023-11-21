"""Tests for the RDKit scorer"""
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from pytest import fixture, mark
from sklearn.pipeline import Pipeline

from examol.score.rdkit.descriptors import compute_doan_2020_fingerprints
from examol.score.rdkit import make_knn_model, RDKitScorer, make_gpr_model
from examol.score.utils.multifi import collect_outputs


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
@mark.parametrize('n_jobs', [1, 2])
def test_functions(training_set, scorer, pipeline, recipe, bootstrap, n_jobs):
    model_msg = scorer.prepare_message(pipeline)
    assert isinstance(model_msg, Pipeline)
    pipeline.steps[0][1].n_jobs = n_jobs

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


@mark.parametrize('pre_compute', [True, False])
@mark.parametrize('num_pcs', [None, 2])
def test_gpr(training_set, scorer, recipe, num_pcs, pre_compute):
    pipeline = make_gpr_model(num_pcs=num_pcs)

    # Set up for pre-computing
    if pre_compute:
        scorer = RDKitScorer(pre_transform=pipeline.steps[0][1])
        pipeline.steps.pop(0)

    # Test training
    model_msg = scorer.prepare_message(pipeline)
    inputs = scorer.transform_inputs(training_set)
    outputs = scorer.transform_outputs(training_set, recipe)
    update_msg = scorer.retrain(model_msg, inputs, outputs, bootstrap=False)
    pipeline = scorer.update(pipeline, update_msg)

    if num_pcs is None and not pre_compute:
        assert pipeline.steps[1][1].best_estimator_.steps[0][1].n_components < 10


@mark.parametrize('bootstrap', [False, True])
@mark.parametrize('actually_single', [False, True])
def test_multifi(training_set, multifi_recipes, scorer, pipeline, bootstrap, actually_single):
    # Emulate what happens if we don't have any steps
    if actually_single:
        multifi_recipes = multifi_recipes[:1]

    # Test conversion to multi-fidelity
    inputs = scorer.transform_inputs(training_set)
    lower_fidelities = collect_outputs(training_set, multifi_recipes[:-1])

    # Test training
    model_msg = scorer.prepare_message(pipeline, training=True)
    outputs = scorer.transform_outputs(training_set, multifi_recipes[-1])
    update_msg = scorer.retrain(model_msg, inputs, outputs, lower_fidelities=lower_fidelities, bootstrap=bootstrap)
    assert len(update_msg) == len(multifi_recipes)

    # Test updating
    pipeline = scorer.update(pipeline, update_msg)
    assert len(pipeline) == len(multifi_recipes)

    # Test inference
    model_msg = scorer.prepare_message(pipeline, training=False)
    predictions = scorer.score(model_msg, inputs, lower_fidelities=lower_fidelities)
    assert predictions.shape == (len(training_set),)
    if not (bootstrap or actually_single):
        assert np.isclose(predictions, outputs).all()  # Should give exact result, since all values are known

    predictions = scorer.score(model_msg, inputs)
    assert predictions.shape == (len(training_set),)
    if not (bootstrap or actually_single):
        assert np.isclose(predictions, outputs).all()  # Should give exact result, since all values are known and we're using a KNN
