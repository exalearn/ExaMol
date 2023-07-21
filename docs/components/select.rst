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
     - Multi-Objective
     - Batch Aware
     - Multi-Fidelity
   * - :class:`~examol.select.baseline.RandomSelector`
     - Baseline
     - ✔
     - ✘
     - ✘
   * - :class:`~examol.select.baseline.GreedySelector`
     - Baseline
     - ✘
     - ✘
     - ✘
   * - :class:`~examol.select.bayes.ExpectedImprovement`
     - Bayesian
     - ✘
     - ✘
     - ✘
   * - :class:`~examol.select.botorch.BOTorchSequentialSelector`
     - Bayesian*
     - ✘
     - ✘
     - ✘
   * - :class:`~examol.select.botorch.EHVISelector`
     - Bayesian
     - ✔
     - ✘
     - ✘

\* Interface to many classes of selection algorithms


Using a Selector
----------------

Selectors employ a batching strategy to work with very large search spaces.

Start the selection process by creating the Selector then
updating it with the current database and objective functions (i.e., recipes).

.. code-block:: python

    selector = GreedySelector(to_select=2, maximize=True)
    selector.update(database, [recipe])

The Selector can then receive new predictions as a list of "keys" that define which computation
associated with a list of of predictions from an ensemble of machine learning models.


.. code-block:: python

    # Samples array is: <num objectives> x <num samples> x <num models>
    selector.add_possibilities(keys=[1, 2, 3], samples=np.arange(3)[None, :, None])

Continue to add new possibilities until ready to select new computations,
which are retrieved by the "dispense" function.

.. code-block:: python

    print(list(selector.dispense())) # [(3, 3.), (2, 2.)]

Call :meth:`~examol.select.base.Selector.start_gathering` when done dispensing
to clear any remaining results or
call :meth:`~examol.select.base.Selector.add_possibilities` to start adding new possibilities again.
