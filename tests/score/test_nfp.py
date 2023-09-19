import pickle as pkl

from examol.score.base import collect_outputs

try:
    import tensorflow as tf
    from examol.score.nfp import convert_string_to_dict, make_simple_network, NFPScorer, make_data_loader, NFPMessage
except ImportError:
    has_tf = False
else:
    has_tf = True
import numpy as np

from pytest import raises, fixture, mark


@fixture()
def model() -> 'tf.keras.Model':
    return make_simple_network(atom_features=8, message_steps=4, output_layers=[32, 16])


@fixture()
def scorer() -> 'NFPScorer':
    return NFPScorer()


@mark.skipif(not has_tf, reason='TF is not installed')
def test_convert():
    mol_dict = convert_string_to_dict('C')
    assert mol_dict['atom'] == [6, 1, 1, 1, 1]
    assert mol_dict['bond'] == [3] * 8  # Bonds are bidirectional
    assert (mol_dict['connectivity'] == [[0, 1], [0, 2], [0, 3], [0, 4],
                                         [1, 0], [2, 0], [3, 0], [4, 0]]).all()

    # Make sure we fail with unconnected atoms
    with raises(ValueError) as error:
        convert_string_to_dict('N#N.[He]')
    assert 'unconnected atoms' in str(error.value)


@mark.skipif(not has_tf, reason='TF is not installed')
def test_parse_inputs(scorer, training_set):
    parsed_inputs = scorer.transform_inputs(training_set)
    assert len(parsed_inputs) == len(training_set)
    assert isinstance(parsed_inputs[0], dict)


@mark.skipif(not has_tf, reason='TF is not installed')
def test_data_loader(scorer, training_set, model, recipe):
    # Make the dictionary inputs
    parsed_inputs = scorer.transform_inputs(training_set)
    values = scorer.transform_outputs(training_set, recipe)

    # Default options
    loader = make_data_loader(parsed_inputs)
    outputs = model.predict(loader, verbose=False)
    assert np.size(outputs) == 3  # Make sure it parses correctly

    # Make sure it produces both training values
    loader = make_data_loader(parsed_inputs, values)
    x, y = next(iter(loader))
    assert x['atom'].shape[0] == 3
    assert np.size(y) == 3

    # Add a shuffle buffer
    loader = make_data_loader(parsed_inputs, values, shuffle_buffer=128, repeat=True)
    gen = iter(loader)
    _, batch_0 = next(gen)
    _, batch_1 = next(gen)
    assert batch_0.shape == (32,)
    assert not np.isclose(batch_0, batch_1).all()

    # Ensure we drop the last batch
    loader = make_data_loader(parsed_inputs, values, batch_size=2, drop_last_batch=True)
    gen = iter(loader)
    next(gen)
    with raises(StopIteration):
        next(gen)

    # Make sure we pad correctly
    loader = make_data_loader(parsed_inputs, values)
    x, _ = next(iter(loader))

    assert x['atom'].shape == (3, 11)  # CCC has 11 atoms
    assert x['bond'].shape == (3, 20)  # CCC has 10 bonds (x2 for bidirectional)

    loader = make_data_loader(parsed_inputs[:2], values)
    x, _ = next(iter(loader))

    assert x['atom'].shape == (2, 8)  # CC has 8 atoms
    assert x['bond'].shape == (2, 14)  # CC has 7 bonds (x2 for bidirectional)


@mark.parametrize('atomwise', [True, False])
@mark.skipif(not has_tf, reason='TF is not installed')
def test_padded_outputs(atomwise: bool, training_set, scorer):
    model = make_simple_network(8, message_steps=4, output_layers=[32], atomwise=atomwise)

    # Make the dictionary inputs
    parsed_inputs = scorer.transform_inputs(training_set)

    # Run it with all 3 molecules
    loader = make_data_loader(parsed_inputs)
    outputs_3 = model.predict(loader, verbose=False, workers=0)

    # Run it with only the first 2, which will result in a different padding size
    loader = make_data_loader(parsed_inputs[:2])
    outputs_2 = model.predict(loader, verbose=False, workers=0)

    assert np.isclose(outputs_2, outputs_3[:2, :]).all()


@mark.skipif(not has_tf, reason='TF is not installed')
def test_score(model, scorer, training_set):
    parsed_inputs = scorer.transform_inputs(training_set)
    model_msg = scorer.prepare_message(model)

    outputs = scorer.score(model_msg, parsed_inputs)
    assert len(outputs) == len(training_set)


@mark.skipif(not has_tf, reason='TF is not installed')
@mark.parametrize('retrain', [True, False])
def test_train(model, scorer: 'NFPScorer', training_set, retrain: bool, recipe):
    scorer.retrain_from_scratch = retrain
    parsed_inputs = scorer.transform_inputs(training_set)
    parsed_outputs = scorer.transform_outputs(training_set, recipe)
    model_msg = scorer.prepare_message(model, training=True)

    update_msg = scorer.retrain(model_msg, parsed_inputs, parsed_outputs, timeout=1, batch_size=1)
    assert len(update_msg) == 2  # Weights and log

    scorer.update(model, update_msg)


@mark.skipif(not has_tf, reason='TF is not installed')
def test_message(model):
    """Make sure we can serialize"""
    # Create the message
    msg = NFPMessage(model)
    assert msg._model is not None  # At this point, we have a cached copy

    # Serialize it and make sure it reconstitutes
    copied_msg: NFPMessage = pkl.loads(pkl.dumps(msg))
    assert copied_msg._model is None  # Should not have rebuilt the model yet

    copied_model = copied_msg.get_model()
    assert np.isclose(copied_model.weights[0], model.weights[0]).all()


def test_multifi_loader(training_set, scorer, multifi_recipes):
    # Convert the input data
    inputs = scorer.transform_inputs(training_set, multifi_recipes)
    outputs = scorer.transform_outputs(training_set, multifi_recipes)
    assert np.isclose(inputs[0][1], [1., 1.]).all()
    assert np.isclose(outputs[0], inputs[0][1]).all()

    # Test the loader
    input_dicts, _ = zip(*inputs)  # Get just the input dictionaries
    loader = make_data_loader(input_dicts, outputs.tolist(), batch_size=1, value_spec=tf.TensorSpec((2,), dtype=tf.float32))
    batch_x, batch_y = next(iter(loader))
    assert np.isclose(batch_y, [1., 1.]).all()


@mark.parametrize('atomwise', [True, False])
def test_multifi_model(atomwise, training_set, multifi_recipes, scorer):
    # Make the network
    model = make_simple_network(atom_features=8, message_steps=4, output_layers=[32, 16], outputs=2)
    assert model.output_shape == (None, 2)

    # Train it
    parsed_inputs = scorer.transform_inputs(training_set, multifi_recipes)
    parsed_outputs = scorer.transform_outputs(training_set, multifi_recipes)
    model_msg = scorer.prepare_message(model, training=True)

    update_msg = scorer.retrain(model_msg, parsed_inputs, parsed_outputs, timeout=1, batch_size=1)
    model = scorer.update(model, update_msg)

    # Run it
    model_msg = scorer.prepare_message(model, training=False)
    preds = scorer.score(model_msg, parsed_inputs)
    assert np.isfinite(preds).all()
    assert np.isclose(preds, collect_outputs(training_set, multifi_recipes)).all()  # We should not actually use model predictions, so result should be the same
