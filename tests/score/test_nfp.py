import numpy as np
from pytest import raises, fixture, mark
import tensorflow as tf

from examol.score.nfp import convert_string_to_dict, make_simple_network, NFPScorer, make_data_loader


@fixture()
def model() -> tf.keras.Model:
    return make_simple_network(atom_features=8, message_steps=4, output_layers=[32, 16])


@fixture()
def scorer(recipe) -> NFPScorer:
    return NFPScorer(recipe)


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


def test_parse_inputs(scorer, training_set):
    parsed_inputs = scorer.transform_inputs(training_set)
    assert len(parsed_inputs) == len(training_set)
    assert isinstance(parsed_inputs[0], dict)


def test_data_loader(scorer, training_set, model):
    # Make the dictionary inputs
    parsed_inputs = scorer.transform_inputs(training_set)
    values = scorer.transform_outputs(training_set)

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
    loader = make_data_loader(parsed_inputs, values, max_size=64)
    x, _ = next(iter(loader))

    assert x['atom'].shape == (3, 64)
    assert x['bond'].shape == (3, 64 * 4)


@mark.parametrize('atomwise', [True, False])
def test_padded_outputs(atomwise: bool, training_set, scorer):
    model = make_simple_network(8, message_steps=4, output_layers=[32], atomwise=atomwise)

    # Make the dictionary inputs
    parsed_inputs = scorer.transform_inputs(training_set)

    # Run it with padded arrays
    loader = make_data_loader(parsed_inputs)
    outputs_nopad = model.predict(loader, verbose=False)

    # Run it without
    loader = make_data_loader(parsed_inputs, max_size=64)
    outputs_pad = model.predict(loader, verbose=False)

    assert np.isclose(outputs_pad, outputs_nopad).all()
