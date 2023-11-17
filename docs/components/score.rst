Score
=====

The Score module defines interfaces for running machine learning (ML) tasks on distributed systems.
Each implementation of :class:`~examol.score.base.Scorer` provides tools for sending models to
remote compute nodes,
preparing molecular data for training or inference,
and functions for executing training and inference on remote nodes.

Available Interfaces
--------------------

ExaMol provides interfaces to several libraries which support ML on molecular property data.

.. list-table::
    :header-rows: 1

    * - Interface
      - Model Types
      - Description
    * - :class:`~examol.score.rdkit.RDKitScorer`
      - Conventional ML
      - Models which use fingerprints computed from RDKit as inputs to scikit-learn Pipelines.
    * - :class:`~examol.score.nfp.NFPScorer`
      - MPNNs
      - Neural networks based on the `Neural Fingerprints (nfp) library <https://github.com/NREL/nfp>`_,
        which is backed by Tensorflow

Modules for each type of learning algorithms provide helper functions to generate models.
For example, :py:meth:`~examol.score.rdkit.make_knn_model` creates a KNN model.

Using Scorers
-------------

Scorers separate pre-processing data, transmitting models, and running ML tasks into separate steps
so that they can be distributed across supercomputing resources.

Consider model training as an example.
Start by creating a scorer, a model it will train, and the recipe describing the computations to be learned.

.. code-block:: python

    scorer = RDKitScorer()
    recipe = RedoxEnergy(charge=1, config_name='xtb')
    model = make_knn_model()

Training the model requires first transforming the available molecule data
(as `molecule data records <store.html#data-models>`_)
into inputs and outputs compatible with the scorer.

.. code-block:: python

    outputs = model.transform_outputs(records, recipe)  # Outputs are specific to a recipe
    inputs = model.transform_inputs(records)  # Inputs are not

Then, convert the model into a form that can be transmitted across nodes

.. code-block:: python

    model_msg = model.prepare_message(model, training=True)

ExaMol is now ready to run training on a remote node, and will use the output of training to update the local
copy of the model:

.. code-block:: python

    update_msg = scorer.retrain(model_msg, inputs, outputs)  # Can be run remotely
    model = scorer.update(model, update_msg)

Multi-fidelity Learning
-----------------------

Some Scorer classes support using properties computed at lower levels of accuracy
to improve performance.
The strategies employed by each Scorer may be different, but all have the same interface.

Use the multi-fidelity capability of a Scorer by providing
values from lower levels of fidelity when training or running inference.

.. code-block:: python

    from examol.score.utils.multifi import collect_outputs
    fidelities = [RedoxEnergy(1, 'low'), RedoxEnergy(1, 'medium'), RedoxEnergy(1, 'high')]

    # Get the inputs and outputs, as normal
    inputs = scorer.transform_inputs(records)
    outputs = scorer.transform_outputs(records, fidelities[-1])  # Train using the highest level

    # Pass the low-fidelity results to scoring and inference
    lower_fidelities = collect_outputs(records, fidelities[:-1])
    scorer.train(model_msg, inputs, outputs, lower_fidelties=lower_fidelities)
    ...
    scorer.score(model_msg, inputs, lower_fidelties=lower_fidelities)
