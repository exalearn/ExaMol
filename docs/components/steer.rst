Steer
=====

ExaMol scales to use large supercomputers by managing many tasks together.
The logic for when to launch tasks and how to process completed tasks are defined
as `Colmena <https://colmena.readthedocs.io/>`_ "Thinker" classes.
ExaMol contains several different Thinkers, which each use different strategies
for deploying tasks on a supercomputer.

Available Methods
-----------------

Each steering strategy is associated with a specific `Solution strategy <solution.html>`_.

.. list-table::
    :header-rows: 1

    * - Class
      - Solution
      - Description
    * - :class:`~examol.steer.baseline.BruteForceThinker`
      - :class:`~examol.specify.base.SolutionSpecification`
      - Evaluate all molecules in an initial population
    * - :class:`~examol.steer.single.SingleStepThinker`
      - :class:`~examol.solution.SingleFidelityActiveLearning`
      - Run all recipes for each selected molecule
    * - :class:`~examol.steer.multifi.PipelineThinker`
      - :class:`~examol.solution.MultiFidelityActiveLearning`
      - Run the next step in a pipeline each time a model is selected


Single Objective Thinker as an Example
--------------------------------------

The :class:`~examol.steer.single.SingleStepThinker` is a good example for explaining how Thinkers work in ExaMol.

The strategy for this thinker is three parts:

#. Never leave nodes on the supercomputer idle
#. Update the list of selected calculations with new data as quickly as possible
#. Wait until resources are free until submitting the next calculation.

This strategy is achieved by a series of simple policies, such as:

- Submit a new quantum chemistry calculation when another completes
- Begin re-training models as soon as a recipe is complete for any molecule
- Re-run inference for all molecules as soon as all models finish training

These policy steps are defined as methods of the Thinker marked with a special decorator
(see `Colmena's quickstart <https://colmena.readthedocs.io/en/latest/quickstart.html>`_).
For example, the "submit a new quantum chemistry" policy is defined by a pair of methods

.. code-block:: python

    class SingleStepThinker(MoleculeThinker):
        ...
        @result_processor(topic='simulation')
        def store_simulation(self, result: Result):
            """Store the output of a simulation"""
            # Trigger a new simulation to start
            self.rec.release()
            ...

        @task_submitter()
        def submit_simulation(self):
            """Submit a simulation task when resources are available"""
            record, suggestion = next(self.task_iterator)  # Returns a molecule record and the suggested computation
            ...



``store_simulation``, runs when a simulation result completes
and starts by marking resources available before updating the database
and - if conditions are right - retraining the models.
``submit_simulation`` is started as soon as resources are marked as free,
keeping the supercomputer occupied.

The other methods manage keeping machine learning models up-to-date and
ensuring the task iterator (``self.task_iterator``) produces the best possible computations to run.
