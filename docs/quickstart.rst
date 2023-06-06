Quickstart
==========

An ExaMol application is divided into a _Thinker_ which defines which tasks to run
and a _Doer_ which executes them on HPC resources.
Your task is to define the Thinker and Doer by a Python specification object.

The specification object, :class:`~examol.specify.ExaMolSpecification`,
describes what thinker is seeking to optimize,
how it will select calculations,
what those computations are,
and a description of the resources available to it.
A simple example looks something like:

.. code-block:: python

    recipe = RedoxEnergy(charge=1, compute_config='xtb')  # What we're trying to optimize
    spec = ExaMolSpecification(
        database='training-data.json',
        recipe=recipe,
        search_space='search_space.smi',
        selector=GreedySelector(n_to_select=8, maximize=True),
        simulator=ASESimulator(scratch_dir='/tmp'),
        scorer=RDKitScorer(recipe),
        models=[KNeighborsRegressor()],
        num_to_run=8,
        thinker=SingleObjectiveThinker,
        compute_config=config,
        run_dir='run'
    )

We'll go through each option briefly here,
and link out to pages that describe the full options available for each.

What and How to Compute
-----------------------

The ``recipe`` and ``simulator`` options define which molecule property to compute
and an interface for ExaMol to compute it, respectively.

Both recipes and simulator are designed to ensure all calculations in a set are performed with consistent settings.
ExaMol defines a set of pre-defined levels of accuracies, which are enumerated in
`the Simulate documentation <components/simulate.html#levels>`_.

Recipes are based on the :class:`~examol.store.recipes.PropertyRecipe` class,
and implement methods to compute a certain property and determine which computations are needed.
Your specification will contain the details of what you wish to compute (e.g., which solvent for a solvation energy)
and the level of accuracy to compute it (e.g., which XC functional)?
See the list recipes and learn how to make your own `in the component documentation <components/store.html#recipes>`_.

The simulator is based on :class:`~examol.simulate.BaseSimulator` class and
defines an interface to the computational chemistry code used to assess molecular energies.
Your specification will contain information on how to run each supported code on a specific supercomputer,
such as the path to its executable and how many nodes to use for each task.
See how to create one in the `Simulate documentation <components/simulate.html#the-simulator-interface>`_.

Starting Data
-------------

The starting data for a project is a line-delimited JSON describing what molecular properties are already known.
Each line of the file is a different molecule, with data following the :class:`~examol.store.models.MoleculeRecord` format.

We recommend creating the initial database by running a seed set of molecules with a purpose-built scripts.

.. note:: I'll upload some example scripts soon.

