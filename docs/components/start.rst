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

Using a Starter
---------------

Define a :class:`~examol.start.base.Starter` methods by setting dataset size under which it will be run
(``threshold``) and how many it will select (``to_select``), and the maximum number of molecules in the
search space to consider (``max_to_consider``).

The threshold and selection size are different because so that you can select enough molecules
to fill a supercomputer fully even if the database is already close to the threshold size.

There is an (optional) threshold on the size of molecules to consider as ExaMol is intended to be used
for enormous search spaces.

Once defined, provide an iterator over the names of molecules to consider:

.. code-block:: python

    starter = RandomStarter(threshold=4, to_select=2)
    starting_pool = starter.select(['C', 'O', 'N'])
