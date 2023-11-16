"""Scheduling strategies for multi-fidelity design campaigns"""
from pathlib import Path
from multiprocessing import Pool
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
from examol.utils.chemistry import get_inchi_key_from_molecule_string


class PipelineThinker(ScorerThinker):
    """Thinker which runs each level of fidelity in incremental steps

    See :class:`~examol.solution.MultiFidelityActiveLearning` for a description
    of the adjustable parameters.
    """

    solution: MultiFidelityActiveLearning

    already_in_db: set[str]
    """InChI keys of molecules from the search space which are already in the database"""

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

        # Initialize the list of relevant database records
        self.already_in_db = self.get_relevant_database_records()

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
        self.already_in_db.add(record.key)  # A record gets created above
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

    def _simulations_complete(self, record: MoleculeRecord):
        super()._simulations_complete(record)
        self.already_in_db.add(record.key)  # Make sure it is there

    def get_relevant_database_records(self) -> set[str]:
        """Get only the entries from the database which are in the search space

        Returns:
            InChI keys from the database which are in the search space
        """

        # Get the database of all keys
        matched = set()
        all_keys = set(r.key for r in self.database.iterate_over_records())
        if len(all_keys) == 0:
            return matched

        # Evaluate against molecules from the search spaces in batches
        self.logger.info(f'Searching for {len(all_keys)} molecules from the database in our search space')
        with Pool(4) as pool:
            for search_key in pool.imap_unordered(get_inchi_key_from_molecule_string, self.iterate_over_search_space(only_smiles=True), chunksize=10000):
                if search_key in all_keys:
                    matched.add(search_key)
                    all_keys.remove(search_key)

        return matched
