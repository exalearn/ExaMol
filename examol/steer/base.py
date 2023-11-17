"""Base class that defines core routines used across many steering policies"""
import gzip
import json
import logging
from pathlib import Path
from dataclasses import asdict
from threading import Condition
from collections import defaultdict
from typing import Iterator, Sequence, Iterable

import numpy as np
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, ResourceCounter, result_processor, task_submitter

from examol.simulate.base import SimResult
from examol.solution import SolutionSpecification
from examol.store.db.base import MoleculeStore
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe, SimulationRequest


class MoleculeThinker(BaseThinker):
    """Base for a thinker which performs molecular design

    Args:
        queues: Queues used to communicate with the task server
        rec: Counter used to track availability of different resources
        run_dir: Directory in which to store results
        recipes: List of recipes to compute
        solution: Description of how to solve the problem
        database: List of molecule records
        search_space: Lists of molecules to be evaluated as a list of ".smi" or ".json" files
    """

    database: MoleculeStore
    """Access to the data available to the thinker"""

    task_queue: list[tuple[str, float]]
    """List of tasks to run. Each entry is a SMILES string and score, and they are arranged descending in priority"""
    task_queue_lock: Condition
    """Lock used to control access to :attr:`task_queue`"""

    def __init__(self,
                 queues: ColmenaQueues,
                 rec: ResourceCounter,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 solution: SolutionSpecification,
                 search_space: list[Path | str],
                 database: MoleculeStore):
        super().__init__(queues, resource_counter=rec)
        self.database = database
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.search_space = search_space

        # Mark where the logs should be stored
        handler = logging.FileHandler(self.run_dir / 'run.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for logger in [self.logger, logging.getLogger('colmena'), logging.getLogger('proxystore')]:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        # Track progress
        self.solution = solution
        self.num_to_run: int = solution.num_to_run
        self.completed: int = 0
        self.molecules_in_progress: dict[str, int] = defaultdict(int)  # Map of InChI Key -> number of ongoing computations

        # Attributes related to simulation
        self.recipes = tuple(recipes)
        self.task_queue_lock = Condition()
        self.task_queue = []  # List of tasks to run, SMILES string and score
        self.task_iterator = self.task_iterator()  # Tool for pulling from the task queue
        self.recipe_types = dict((r.name, r) for r in recipes)

    def iterate_over_search_space(self, only_smiles: bool = False) -> Iterator[MoleculeRecord | str]:
        """Function to produce a stream of molecules from the input files

        Args:
            only_smiles: Whether to return only the SMILES string rather than the full record
        Yields:
            A :class:`MoleculeRecord` for each molecule in the search space or just the SMILES String
        """
        for i, path in enumerate(self.search_space):
            path = Path(path).resolve()
            self.logger.info(f'Reading molecules from file {i + 1}/{len(self.search_space)}: {path.resolve()}')

            # Determine how to read molecules out of the file
            filename_lower = path.name.lower()
            if any(filename_lower.endswith(ext) or filename_lower.endswith(f'{ext}.gz') for ext in ['.smi', '.json']):
                # Open with GZIP or normally depending on the extension
                is_json = '.json' in filename_lower
                with (gzip.open(path, 'rt') if filename_lower.endswith('.gz') else path.open()) as fmols:
                    for line in fmols:
                        if only_smiles and is_json:
                            yield json.loads(line)['identifier']['smiles']
                        elif only_smiles and not is_json:
                            yield line.strip()
                        elif is_json:
                            yield MoleculeRecord.parse_raw(line)
                        else:
                            yield MoleculeRecord.from_identifier(line.strip())
            else:
                raise ValueError(f'File type is unrecognized for {path}')

    def _write_result(self, result: Result, result_type: str):
        with (self.run_dir / f'{result_type}-results.json').open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)

    def _get_next_tasks(self) -> tuple[MoleculeRecord, float, Iterable[PropertyRecipe]]:
        """Get the next task from the task queue

        Assumes that the task queue is locked and there are tasks in the queue
        """
        # Return the next one off the list
        smiles, score = self.task_queue.pop(0)  # Get the next task
        return self.database.get_or_make_record(smiles), score, self.recipes

    def task_iterator(self) -> Iterator[tuple[MoleculeRecord, Iterable[PropertyRecipe], SimulationRequest]]:
        """Iterate over the next tasks in the task queue

        Yields:
            - Molecule being processed
            - Recipes being computed
            - Simulation to execute
        """

        while True:
            # Get the next task to run
            with self.task_queue_lock:
                if len(self.task_queue) == 0:
                    self.logger.info('No tasks available to run. Waiting')
                    while not self.task_queue_lock.wait(timeout=2):
                        if self.done.is_set():
                            yield None, None
            record, score, recipes = self._get_next_tasks()
            self.logger.info(f'Selected {record.key} to run next. Score={score:.2f}, queue length={len(self.task_queue)}')

            # Determine which computations to run next
            try:
                suggestions = set()
                for recipe in recipes:
                    suggestions = set(recipe.suggest_computations(record))
            except ValueError as exc:
                self.logger.warning(f'Generating computations for {record.key} failed. Skipping. Reason: {exc}')
                continue
            self.logger.info(f'Found {len(suggestions)} more computations to do for {record.key}')
            self.molecules_in_progress[record.key] += len(suggestions)  # Update the number of computations in progress for this molecule

            for suggestion in suggestions:
                yield record, recipes, suggestion

    def _simulations_complete(self, record: MoleculeRecord):
        """This function is called when all ongoing computations for a molecule have finished

        Args:
            record: Record for the molecule which had completed
        """
        pass

    @result_processor(topic='simulation')
    def store_simulation(self, result: Result):
        """Store the output of a simulation"""
        # Trigger a new simulation to start
        self.rec.release()

        # Get the molecule record
        mol_key = result.task_info["key"]
        record = self.database[mol_key]
        self.logger.info(f'Received a result for {mol_key}. Runtime={(result.time_running or np.nan):.1f}s, success={result.success}')

        # Update the number of computations in progress
        self.molecules_in_progress[mol_key] -= 1

        # Add our result, see if finished
        if result.success:
            results: list[SimResult]
            if result.method == 'optimize_structure':
                sim_result, steps, metadata = result.value
                results = [sim_result] + steps
                record.add_energies(sim_result, steps)
            elif result.method == 'compute_energy':
                sim_result, metadata = result.value
                results = [sim_result]
                record.add_energies(sim_result)
            else:
                raise NotImplementedError()

            # Assemble the recipes being complete
            recipes = [
                self.recipe_types[r['name']].from_name(**r) for r in result.task_info['recipes']
            ]
            self.logger.info(f'Checking if we have completed recipes: {", ".join([r.name + "//" + r.level for r in recipes])}')

            # If we can compute then property than we are done
            not_done = sum(recipe.lookup(record, recompute=True) is None for recipe in recipes)
            if not_done == 0:
                # If so, mark that we have finished computing the property
                self.completed += 1
                self.logger.info(f'Finished computing all recipes for {mol_key}. Completed {self.completed}/{self.num_to_run} molecules')
                self.molecules_in_progress.pop(mol_key)  # Remove it from the list
                if self.completed == self.num_to_run:
                    self.logger.info('Done!')
                    self.done.set()

                # Mark that we've finished with all recipes
                result.task_info['status'] = 'finished'
                result.task_info['result'] = [recipe.lookup(record) for recipe in self.recipes]
                self._simulations_complete(record)
            else:
                # If not, see if we need to resubmit to finish the computation
                self.logger.info(f'Finished {len(self.recipes) - not_done}/{len(self.recipes)} recipes for {mol_key}')
                result.task_info['status'] = 'in progress'
                if self.molecules_in_progress[mol_key] == 0:
                    self.logger.info('We must submit new computations for this molecule. Re-adding it to the front of the task queue')
                    with self.task_queue_lock:
                        self.task_queue.insert(0, (record.identifier.smiles, np.inf))
                        self.task_queue_lock.notify_all()

            # Update the record in the store
            self.database.update_record(record)

            # Save the relaxation steps to disk
            with open(self.run_dir / 'simulation-records.json', 'a') as fp:
                for record in results:
                    print(record.json(), file=fp)
        else:
            # Remove molecule from the list of those in progress if no other computations remain
            if self.molecules_in_progress[mol_key] == 0:
                self.molecules_in_progress.pop(mol_key)
                self._simulations_complete(record)

        self._write_result(result, 'simulation')

    @task_submitter()
    def submit_simulation(self):
        """Submit a simulation task when resources are available"""
        record, recipes, suggestion = next(self.task_iterator)
        if record is None:
            return  # The thinker is done

        task_info = {'key': record.key,
                     'recipes': [{'name': r.name, 'level': r.level} for r in recipes],
                     'computation': asdict(suggestion)}
        if suggestion.optimize:
            self.logger.info(f'Optimizing structure for {record.key} with a charge of {suggestion.charge}')
            self.queues.send_inputs(
                record.key, suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent,
                method='optimize_structure',
                topic='simulation',
                task_info=task_info
            )
        else:
            self.logger.info(f'Getting single-point energy for {record.key} with a charge of {suggestion.charge} ' +
                             ('' if suggestion.solvent is None else f'in {suggestion.solvent}'))
            self.queues.send_inputs(
                record.key, suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent,
                method='compute_energy',
                topic='simulation',
                task_info=task_info
            )
