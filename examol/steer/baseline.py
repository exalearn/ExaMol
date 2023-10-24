"""Baseline methods for steering a molecular design campaign"""
from pathlib import Path
from typing import Sequence

import numpy as np
from colmena.queue import ColmenaQueues
from colmena.thinker import agent, ResourceCounter

from examol.specify import SolutionSpecification
from examol.steer.base import MoleculeThinker
from examol.store.db.base import MoleculeStore
from examol.store.recipes.base import PropertyRecipe


class BruteForceThinker(MoleculeThinker):
    """Run the selection of molecules selected in the beginning

    Args:
        queues: Queues used to communicate with the task server
        run_dir: Directory in which to store results
        recipes: List of recipes to compute
        solution: Description of how to solve the problem
        database: List of molecule records
        search_space: Lists of molecules to be evaluated as a list of ".smi" or ".json" files
        num_workers: Number of simulations to run in parallel
        overselection: Additional fraction molecules to select above the number requested by the user.
            Number of molecules will be ``solution.num_to_run * overselection``.
            Used to ensure target number of molecules are evaluated even if some fail.
    """

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 solution: SolutionSpecification,
                 search_space: list[Path | str],
                 database: MoleculeStore,
                 num_workers: int = 1,
                 overselection: float = 0):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, recipes, solution, search_space, database)
        self.overselection = overselection

    @agent(startup=True)
    def startup(self):
        """Pre-populate the database, if needed."""

        # Determine the number of computations to put in queue
        num_to_run = int(self.solution.num_to_run * (1 + self.overselection))
        self.logger.info(f'Selecting a total of {num_to_run} molecules to run')

        subset = self.solution.starter.select(list(self.iterate_over_search_space(only_smiles=True)), self.num_to_run)
        self.logger.info(f'Selected {len(subset)} molecules to run')
        with self.task_queue_lock:
            for key in subset:
                self.task_queue.append((key, np.nan))  # All get the same score
            self.task_queue_lock.notify_all()

    def _simulations_complete(self):
        if len(self.task_queue) == 0:
            self.logger.info('Run out of molecules to run. Exiting')
            self.done.set()
