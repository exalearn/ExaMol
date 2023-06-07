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

Using a Selector
----------------

Selectors employ a batching strategy to work with very large search spaces.

Start the selection process by creating the Selector then signaling that it should prepare to receive batches.

.. code-block:: python

    selector = GreedySelector(to_select=2, maximize=True)
    selector.start_gathering()

The Selector can then receive new predictions as a list of "keys" that define which computation
associated with a list of of predictions from a machine learning model.


.. code-block:: python

    selector.add_possibilities(keys=[1, 2, 3], samples=np.array([[1, 2, 3]]).T)

Retrieve the list of selected computations by stopping the gathering mode then generating them
from the "dispense" function.

.. code-block:: python

    selector.start_dispensing()
    print(list(selector.dispense())) # [(3, 3.), (2, 2.)]