"""Base classes used in specifications"""
from dataclasses import dataclass
from typing import Callable

from examol.start.base import Starter
from examol.start.fast import RandomStarter


@dataclass
class SolutionSpecification:
    """Define the components of a solution to a problem"""

    starter: Starter = RandomStarter(threshold=10)
    """How to initialize the database if too small. Default: Pick a single random molecule"""
    num_to_run: int = ...
    """Number of quantum chemistry computations to perform"""

    def generate_functions(self) -> list[Callable]:
        """Generate functions to be run on compute nodes

        Returns:
            List of functions ready for use in a workflow system
        """
        return []
