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
the minimum number it will select,
and the maximum number of molecules to consider.

The threshold and selection size are different because so that you can select enough molecules
to fill a supercomputer fully even if the database is already close to the threshold size.

There is an (optional) threshold on the size of molecules to consider as ExaMol is intended to be used
for enormous search spaces.

Once defined, provide an iterator over the names of molecules to consider:

.. code-block:: python

    starter = RandomStarter(threshold=4, min_to_select=2)
    starting_pool = starter.select(['C', 'O', 'N'], 1)  # Will generate choices
