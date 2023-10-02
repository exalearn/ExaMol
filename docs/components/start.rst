Start
=====

The Start modules define how to determine which calculations to perform before
there is enough data available to train a machine learning model.

Available Methods
-----------------

ExaMol provides a few different start methods

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

All :class:`~examol.start.base.Starter` methods require setting
the dataset size under which it will be run,
and the maximum number of molecules to consider.

There is an (optional) threshold on the size of molecules to consider as ExaMol is intended to be used
for enormous search spaces.

Once defined, provide an iterator over the names of molecules to consider:

.. code-block:: python

    starter = RandomStarter(threshold=4)
    starting_pool = starter.select(['C', 'O', 'N'], 2)  # Will generate two choices
