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
