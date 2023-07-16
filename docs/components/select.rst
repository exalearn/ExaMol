Select
======

The Select model define adaptive experimental design algorithms
that select the next computations based on predictions from the
`machine learning models <score.html>`_.

Available Selectors
-------------------

ExaMol includes selectors that have a variety of characteristics

.. list-table::
   :header-rows: 1

   * - Selector
     - Category
     - Batch Aware
     - Multi-Fidelity
   * - :class:`~examol.select.baseline.RandomSelector`
     - Baseline
     - ✘
     - ✘
   * - :class:`~examol.select.baseline.GreedySelector`
     - Baseline
     - ✘
     - ✘
   * - :class:`~examol.select.bayes.ExpectedImprovement`
     - Bayesian
     - ✘
     - ✘

Using a Selector
----------------

Selectors employ a batching strategy to work with very large search spaces.

Start the selection process by creating the Selector,
updating it with the current database and objective function (i.e., recipe),
then signaling that it should prepare to receive batches.

.. code-block:: python

    selector = GreedySelector(to_select=2, maximize=True)
    selector.update(database, recipe)
    selector.start_gathering()

The Selector can then receive new predictions as a list of "keys" that define which computation
associated with a list of of predictions from a machine learning model.


.. code-block:: python

    selector.add_possibilities(keys=[1, 2, 3], samples=np.array([[1, 2, 3]]).T)

Retrieve the list of selected computations by the "dispense" function.

.. code-block:: python

    print(list(selector.dispense())) # [(3, 3.), (2, 2.)]

Call ``start_gathering`` again to clear any previous results before
adding new possibilities.