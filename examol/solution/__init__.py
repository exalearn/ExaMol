"""Specifications for different solution methods"""
from functools import partial, update_wrapper
from dataclasses import field, dataclass
from typing import Callable

from examol.score.base import Scorer
from examol.select.base import Selector
from examol.start.base import Starter
from examol.start.fast import RandomStarter


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
    """List of machine learning models used to predict outcome of :attr:`recipes`"""
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
