"""Base class that defines core routines used across many steering policies"""
import os
import gzip
import json
import shutil
import logging
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from dataclasses import asdict
from threading import Condition
from collections import defaultdict
from typing import Iterator, Sequence

import numpy as np
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, ResourceCounter, result_processor, task_submitter
from more_itertools import batched
from proxystore.store import Store, get_store

from examol.score.base import Scorer
from examol.simulate.base import SimResult
from examol.solution import SolutionSpecification
from examol.store.db.base import MoleculeStore
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe, SimulationRequest


def _generate_inputs(record: MoleculeRecord, scorer: Scorer) -> tuple[str, object] | None:
    """Parse a molecule then generate a form ready for inference

    Args:
        record: Molecule record to be parsed
        scorer: Tool used for inference
    Returns:
        - Key for the molecule record
        - Inference-ready format
        Or None if the transformation fails
    """

    try:
        # Compute the features
        readied = scorer.transform_inputs([record])[0]
    except (ValueError, RuntimeError):
        return None
    return record.identifier.smiles, readied


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

    def task_iterator(self) -> Iterator[tuple[MoleculeRecord, SimulationRequest]]:
        """Iterate over the next tasks in the task queue"""

        while True:
            # Get the next molecule to run
            with self.task_queue_lock:
                if len(self.task_queue) == 0:
                    self.logger.info('No tasks available to run. Waiting')
                    while not self.task_queue_lock.wait(timeout=2):
                        if self.done.is_set():
                            yield None, None
                smiles, score = self.task_queue.pop(0)  # Get the next task
                self.logger.info(f'Selected {smiles} to run next. Score={score:.2f}, queue length={len(self.task_queue)}')

            # Get the molecule record
            record = MoleculeRecord.from_identifier(smiles)
            if record.key in self.database:
                record = self.database[record.key]
            else:
                self.database.update_record(record)

            # Determine which computations to run next
            try:
                suggestions = set()
                for recipe in self.recipes:
                    suggestions = set(recipe.suggest_computations(record))
            except ValueError as exc:
                self.logger.warning(f'Generating computations for {smiles} failed. Skipping. Reason: {exc}')
                continue
            self.logger.info(f'Found {len(suggestions)} more computations to do for {smiles}')
            self.molecules_in_progress[record.key] += len(suggestions)  # Update the number of computations in progress for this molecule

            for suggestion in suggestions:
                yield record, suggestion

    def _simulations_complete(self):
        """This function is called when all ongoing computations for a molecule have finished"""
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

            # If we can compute then property than we are done
            not_done = sum(recipe.lookup(record, recompute=True) is None for recipe in self.recipes)  # TODO (wardlt): Keep track of recipe being computed
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
                self._simulations_complete()
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
                self._simulations_complete()

        self._write_result(result, 'simulation')

    @task_submitter()
    def submit_simulation(self):
        """Submit a simulation task when resources are available"""
        record, suggestion = next(self.task_iterator)
        if record is None:
            return  # The thinker is done

        if suggestion.optimize:
            self.logger.info(f'Optimizing structure for {record.key} with a charge of {suggestion.charge}')
            self.queues.send_inputs(
                record.key, suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent,
                method='optimize_structure',
                topic='simulation',
                task_info={'key': record.key, **asdict(suggestion)}
            )
        else:
            self.logger.info(f'Getting single-point energy for {record.key} with a charge of {suggestion.charge} ' +
                             ('' if suggestion.solvent is None else f'in {suggestion.solvent}'))
            self.queues.send_inputs(
                record.key, suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent,
                method='compute_energy',
                topic='simulation',
                task_info={'key': record.key, **asdict(suggestion)}
            )


class ScorerThinker(MoleculeThinker):
    """

    Args:
        queues: Queues used to communicate with the task server
        rec: Tool used to control the number of tasks being deployed on each resource
        run_dir: Directory in which to store logs, etc.
        recipes: Recipes used to compute the target properties
        database: Connection to the store of molecular data
        solution: Settings related to tools used to solve the problem (e.g., active learning strategy)
        search_space: Search space of molecules. Provided as a list of paths to ".smi" files
        inference_chunk_size: Number of molecules to run inference on per task
    """

    search_space_dir: Path
    """Cache directory for search space"""
    search_space_smiles: list[list[str]]
    """SMILES strings of molecules in the search space"""
    search_space_inputs: list[list[object]]
    """Inputs (or proxies of inputs) to the machine learning models for each molecule in the search space"""

    scorer: Scorer
    """Class used to communicate data and models to distributed workers"""

    def __init__(self,
                 queues: ColmenaQueues,
                 rec: ResourceCounter,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 scorer: Scorer,
                 solution: SolutionSpecification,
                 search_space: list[Path | str],
                 database: MoleculeStore,
                 inference_chunk_size: int = 10000):
        super().__init__(queues, rec, run_dir, recipes, solution, search_space, database)
        self.search_space_dir = self.run_dir / 'search-space'
        self.scorer = scorer
        self._cache_search_space(inference_chunk_size, search_space)

        # Partition the search space into smaller chunks
        self.search_space_smiles: list[list[str]]
        self.search_space_inputs: list[list[object]]
        self.search_space_smiles, self.search_space_inputs = zip(*self._cache_search_space(inference_chunk_size, self.search_space))

    def _cache_search_space(self, inference_chunk_size: int, search_space: list[str | Path]):
        """Cache the search space into a directory within the run"""

        # Check if we must rebuild the cache
        rebuild = True
        config_path = self.search_space_dir / 'settings.json'
        my_config = {
            'inference_chunk_size': inference_chunk_size,
            'scorer': str(self.scorer),
            'paths': [str(Path(p).resolve()) for p in search_space]
        }
        if config_path.exists():
            config = json.loads(config_path.read_text())
            rebuild = config != my_config
            if rebuild:
                self.logger.info('Settings have changed. Rebuilding the cache')
                shutil.rmtree(self.search_space_dir)
        elif self.search_space_dir.exists():
            shutil.rmtree(self.search_space_dir)
        self.search_space_dir.mkdir(exist_ok=True, parents=True)

        # Get the paths to inputs and keys, either by rebuilding or reading from disk
        search_space_keys = {}
        if rebuild:
            # Build search space and save to disk

            # Process the inputs and store them to disk
            search_size = 0
            input_func = partial(_generate_inputs, scorer=self.scorer)
            with ProcessPoolExecutor(min(4, os.cpu_count())) as pool:
                mol_iter = pool.map(input_func, self.iterate_over_search_space(), chunksize=1000)
                mol_iter_no_failures = filter(lambda x: x is not None, mol_iter)
                for chunk_id, chunk in enumerate(batched(mol_iter_no_failures, inference_chunk_size)):
                    keys, objects = zip(*chunk)
                    search_size += len(keys)
                    chunk_path = self.search_space_dir / f'chunk-{chunk_id}.pkl.gz'
                    with gzip.open(chunk_path, 'wb') as fp:
                        pkl.dump(objects, fp)

                    search_space_keys[chunk_path.name] = keys
            self.logger.info(f'Saved {search_size} search entries into {len(search_space_keys)} batches')

            # Save the keys and the configuration
            with open(self.search_space_dir / 'keys.json', 'w') as fp:
                json.dump(search_space_keys, fp)
            with config_path.open('w') as fp:
                json.dump(my_config, fp)
        else:
            # Load in keys
            self.logger.info(f'Loading search space from {self.search_space_dir}')
            with open(self.search_space_dir / 'keys.json') as fp:
                search_space_keys = json.load(fp)

        # Load in the molecules, storing them as proxies in the "inference" store if there is a store defined
        self.logger.info(f'Loading in molecules from {len(search_space_keys)} files')
        output = []

        proxy_store = self.inference_store
        if proxy_store is not None:
            self.logger.info(f'Will store inference objects to {proxy_store}')

        for name, keys in search_space_keys.items():
            with gzip.open(self.search_space_dir / name, 'rb') as fp:  # Load from disk
                objects = pkl.load(fp)

            if proxy_store is not None:  # If the store exists, make a proxy
                objects = proxy_store.proxy(objects)
            output.append((keys, objects))
        return output

    @property
    def inference_store(self) -> Store | None:
        """Proxystore used for inference tasks"""
        if (store_name := self.queues.proxystore_name.get('inference')) is not None:
            return get_store(store_name)
