"""Scheduling strategies for multi-fidelity design campaigns"""
from pathlib import Path
from functools import cached_property
from typing import Sequence, Iterable

import numpy as np
from colmena.queue import ColmenaQueues
from colmena.thinker import ResourceCounter

from examol.solution import MultiFidelityActiveLearning
from examol.steer.base import ScorerThinker
from examol.store.db.base import MoleculeStore
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


class PipelineThinker(ScorerThinker):
    """Thinker which runs each level of fidelity in incremental steps

    See :class:`~examol.solution.MultiFidelityActiveLearning` for a description
    of the adjustable parameters.
    """

    solution: MultiFidelityActiveLearning

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 database: MoleculeStore,
                 solution: MultiFidelityActiveLearning,
                 search_space: list[Path | str],
                 num_workers: int = 2,
                 inference_chunk_size: int = 10000):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, recipes, solution.scorer, solution, search_space, database, inference_chunk_size)

    @cached_property
    def steps(self) -> Sequence[Sequence[PropertyRecipe]]:
        output = list(self.solution.steps)
        output.append(self.recipes)
        return tuple(output)

    @cached_property
    def num_levels(self):
        return 1 + len(self.solution.steps)

    def get_level(self, smiles: str) -> int:
        """Get the current step number of a molecule

        Args:
            smiles: SMILES string of molecule in question
        Returns:
            Step level (0 means no data)
        """

        # See which recipes have been completed
        record = self.database.get_or_make_record(smiles)
        for i, recipes in enumerate(self.steps):
            for recipe in recipes:
                if recipe.level not in record.properties.get(recipe.name, {}):
                    return i

        return self.num_levels

    def _get_next_tasks(self) -> tuple[MoleculeRecord, float, Iterable[PropertyRecipe]]:
        # Determine which level of accuracy to run
        weights = np.cumprod([self.solution.pipeline_target] * self.num_levels)
        weights /= weights.sum()
        target_level = np.random.choice(self.num_levels, p=weights)
        if target_level == 0:
            self.logger.info('Running a new molecule for the first step in the pipeline')
        else:
            self.logger.info(f'Finding a molecule which has completed step #{target_level}')

        # Find a molecule at the target level or the one closest to it
        current_best: tuple[int, int] = (-1, 0)  # Level, index
        for ind, (smiles, score) in enumerate(self.task_queue):
            my_level = self.get_level(smiles)
            if my_level == target_level:
                current_best = (my_level, ind)
                break
            elif target_level > my_level > current_best[0]:
                current_best = (my_level, ind)

        # Return the best choice
        chosen_level, chosen_ind = current_best
        smiles, score = self.task_queue.pop(chosen_ind)
        if chosen_level == -1:
            chosen_level = self.get_level(smiles)
        self.logger.info(f'Pulled molecule at position {chosen_ind} to run at level #{chosen_level}')
        return self.database.get_or_make_record(smiles), score, self.steps[chosen_level]
