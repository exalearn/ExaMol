Start
=====

The Start modules define how to determine which calculations to perform before
there is enough data available to train a machine learning model.

Available Methods
-----------------

ExaMol provides a few different start methods, each with a maximum recommended search space size.

.. list-table::
   :header-rows: 1

   * - Starter
     - Category
     - Maximum Search Size
   * - :class:`~examol.start.fast.RandomStarter`
     - Fast
     - 100M
   * - :class:`~examol.start.kmeans.KMeansStarter`
     - KMeans
     - 100K

Using a Starter
---------------

Simply provide an iterator over the names of molecules to consider:

.. code-block:: python

    starter = RandomStarter()
    starting_pool = starter.select(['C', 'O', 'N'], 2)  # Will generate two choices

The starter will provide a list of SMILES strings from those that were provided.

Increase the speed of selection by setting the ``max_to_consider`` option of the Starter,
which will truncate the list of molecules strings at a specific size before running the selection algorithm.