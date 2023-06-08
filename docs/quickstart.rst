Quickstart
==========

Let us consider the
`redoxmer design example <https://github.com/exalearn/ExaMol/tree/main/examples/redoxmers>`_
as a way to learn how to use ExaMol.

.. note::
    This example assumes you installed xTB and other optional dependencies.
    We recommend you install the `CPU version of ExaMol via Anaconda <installation#recommended-anaconda>`_.

Running ExaMol
--------------

Run ExaMol by calling a command-line executable which launches computations across many resources.

The executable takes at least one argument: the path to a Python specification file and the name of the variable
within that file which is the specification object.

.. code-block:: shell

    examol run examples/redoxmers/spec.py:spec

ExaMol will start writing logging messages to the screen to tell you what it is doing,
which is first to execute the specification file and load in the file you want

.. code-block::

    2023-06-08 12:57:10,063 - examol - INFO - Starting ExaMol v0.0.1
    2023-06-08 12:57:11,916 - examol.cli - INFO - Loaded specification from spec.py, where it was named spec

Once loaded, ExaMol will create the functions to be executed (e.g., "run quantum chemistry")
and start a Parsl workflow engine in a subprocess.
The program will then launch `a steering engine <#steering-strategy>`_ in a thread before beginning
a series of monitoring routines as other threads.

ExaMol will continue writing logging message to screen from all of these threads and will exit
once the steering engine completes.

Understanding Outputs
---------------------

All data from an ExaMol run is written to the output directory defined in the specification file.

Common files for the workflow include:

- ``run.log``: The logging messages
- ``*-results.json``: Metadata about each task completed by ExaMol (e.g., if successful, when started) in
  a line-delimited JSON format. Records follow Colmena's :class:`~colmena.models.Result` schema.
- ``database.json``: Data about each molecule assessed by ExaMol where each line follows
  the :class:`~examol.store.models.MoleculeRecord` format.
- ``report.md``: A report of the workflow performance thus far.

The run directory will contain data from all previous runs.

.. note:: The example specification deletes any previous runs, but this is just for demonstration purposes.

Configuring ExaMol
------------------

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
        scorer=RDKitScorer(),
        models=[KNeighborsRegressor()],
        num_to_run=8,
        thinker=SingleObjectiveThinker,
        thinker_options=dict(num_workers=2),
        compute_config=config,
        run_dir='run'
    )

We'll go through each option briefly here,
and link out to pages that describe the full options available for each.

Quantum Chemistry
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~

The starting data for a project is a line-delimited JSON describing what molecular properties are already known.
Each line of the file is a different molecule, with data following the :class:`~examol.store.models.MoleculeRecord` format.

We recommend creating the initial database by running a seed set of molecules with a purpose-built scripts.

.. note:: I'll upload some example scripts soon and describe them here.

Machine Learning
~~~~~~~~~~~~~~~~

ExaMol uses machine learning (ML) to estimate the output of computations.
The specification requires you to define an interface to run machine learning models (``scorer``) and
then a set of models (``models``) to be trained using that interface.

The Scorer, like the `Simulator used in quantum chemistry <#quantum-chemistry>`_, defines an interface
for the ML computations should be configured with information about how to run the model on your resources.
ExaMol provides interfaces for `a few common libraries <components/score.html>`_) used in ML for molecular properties.

The ``models`` define specific architectures used by the scorer.
Each model will be trained using a different subset of the training data,
and the predictions of all models will be combined to produce predictions with uncertainties for each model.

Search Algorithm
~~~~~~~~~~~~~~~~

The design process is defined by the space of molecules (``search_space``),
how to search through them (``selector``),
and how many quantum chemistry computations will be run (``num_to_run``).

The ``search_space`` option requires the path to a list of SMILES strings as a single file.

The selector defines an adaptive experimental design algorithm -- an algorithm which uses the predictions
from machine learning models to identify the best computations.
ExaMol includes `several selection routines <components/select.html#available-selectors>`_.

Steering Strategy
~~~~~~~~~~~~~~~~~

The ``thinker`` provides the core capability behind ExaMol scaling to large supercomputers:
the ability to schedule many different different tasks at once.
A Thinker strategy defines when to submit new tasks and what to do once they complete.
There is only one strategy available in ExaMol right now, :class:`~examol.steer.single.SingleObjectiveThinker`,
but more will become available as we build the library.

Learn more in the `component documentation <components/steer.html>`_.

Computational Resources
~~~~~~~~~~~~~~~~~~~~~~~

``compute_config`` requires a Parsl :class:`~parsl.config.Config` object describing the resources available to ExaMol.
Parsl's `quickstart describes the basics <https://parsl.readthedocs.io/en/stable/quickstart.html>`_ of
how to describe the queueing system and compute nodes of your supercomputer.
