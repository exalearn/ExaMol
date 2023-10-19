Solution
========

The Solution modules of an ExaMol specification contain the components used for different strategies of optimizing a material.
Descriptions of a solution use different ExaMol components (e.g., `Scorer classes <score.html>`_)
and the same solution can be enacted with different `Steering strategies <steer.html>`_.

Available Methods
-----------------

ExaMol provides multiple solution methods, each described using a different class.

.. list-table::
    :header-rows: 1

    * - Class
      - Description
    * - :class:`~examol.specify.base.SolutionSpecification`
      - The base strategy. Lacks any strategy for using new data to select the next computations.
    * - :class:`~examol.specify.solution.SingleFidelityActiveLearning`
      - Use predictions from machine learning models to select the next computation.
