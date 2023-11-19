"""Specifications for different solution methods"""
from functools import partial, update_wrapper
from dataclasses import field, dataclass
from typing import Callable, Sequence, Collection

from examol.score.base import Scorer
from examol.select.base import Selector
from examol.start.base import Starter
from examol.start.fast import RandomStarter
from examol.store.recipes import PropertyRecipe


@dataclass
class SolutionSpecification:
    """Define the components of a solution to a problem"""

    starter: Starter = RandomStarter()
    """How to initialize the database if too small. Default: Pick a single random molecule"""
    num_to_run: int = ...
    """Number of quantum chemistry computations to perform"""

    def generate_functions(self) -> list[Callable]:
        """Generate functions to be run on compute nodes

        Returns:
            List of functions ready for use in a workflow system
        """
        return []


@dataclass
class SingleFidelityActiveLearning(SolutionSpecification):
    """Tools needed to solve a multi-objective problem using active learning"""

    # Components of the solution
    selector: Selector = ...
    """How to identify which computation to perform next"""
    scorer: Scorer = ...  # TODO (wardlt): Support a different type of model for each recipe
    """Defines algorithms used to retrain and run :attr:`models`"""
    models: list[list[object]] = ...
    """List of machine learning models used to predict outcome of :attr:`recipes`.

    ``models[i][j] is model ``j`` for property ``i``"""
    minimum_training_size: int = 10
    """Minimum database size before starting training. Thinkers will run selections from :attr:`starter` before them"""

    # Options for key operations
    train_options: dict = field(default_factory=dict)
    """Options passed to the :py:meth:`~examol.score.base.Scorer.retrain` function"""
    score_options: dict = field(default_factory=dict)
    """Options passed to the :py:meth:`~examol.score.base.Scorer.score` function"""

    def generate_functions(self) -> list[Callable]:
        def _wrap_function(fun, options: dict):
            wrapped_fun = partial(fun, **options)
            update_wrapper(wrapped_fun, fun)
            return wrapped_fun

        return [
            _wrap_function(self.scorer.retrain, self.train_options),
            _wrap_function(self.scorer.score, self.score_options)
        ]


@dataclass
class MultiFidelityActiveLearning(SingleFidelityActiveLearning):
    """Tools needed for solving a multi-fidelity active learning problem

    Users must define a series of recipes to be run at each step in the workflow,
    :attr:`steps`.
    The next step is run each time a molecule is selected for execution.
    For example, a Thinker would run all recipes in the first step for a molecule for which no data is available
    and then the second step of recipes after all from the first have completed.

    The user also specifies a set fraction of entries to progress through each stage,
    which sets the probability of selecting a certain step in the calculation.
    For example a :attr:`pipeline_target` of 0.1 means that 10% of entries will
    pass through each stage of the pipeline.
    We can achieve this target by selecting to run the first stage of the pipeline
    10 times more often then the next stage.
    """

    steps: Sequence[Collection[PropertyRecipe]] = ()
    """Incremental steps to perform along the way to a maximum level of fidelity"""

    pipeline_target: float = 0.1
    """Fraction of entries to progress through each stage of the pipeline"""

    def get_levels_for_property(self, recipe: PropertyRecipe) -> list[PropertyRecipe]:
        """Get the list of levels at which we compute a certain property"""

        levels = []
        for recipes in self.steps:
            for step in recipes:
                if recipe.name == step.name:
                    levels.append(step)
        levels.append(recipe)
        return levels
