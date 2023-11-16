"""Single-objective and single-fidelity implementation of active learning. As easy as we get"""
import os
import gzip
import json
import pickle as pkl
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from queue import Queue
from threading import Event
from time import perf_counter
from typing import Sequence

import numpy as np
from colmena.proxy import get_store
from colmena.queue import ColmenaQueues
from colmena.thinker import event_responder, ResourceCounter, agent
from more_itertools import interleave_longest, batched
from proxystore.proxy import extract, Proxy
from proxystore.store import Store
from proxystore.store.utils import get_key, ConnectorKeyT

from .base import MoleculeThinker
from examol.solution import SingleFidelityActiveLearning
from ..score.base import Scorer
from ..store.db.base import MoleculeStore
from ..store.models import MoleculeRecord
from ..store.recipes import PropertyRecipe


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


class SingleStepThinker(MoleculeThinker):
    """A thinker which submits all computations needed to evaluate a molecule whenever it is selected

    Args:
        queues: Queues used to communicate with the task server
        run_dir: Directory in which to store logs, etc.
        recipes: Recipes used to compute the target properties
        database: Connection to the store of molecular data
        solution: Settings related to tools used to solve the problem (e.g., active learning strategy)
        search_space: Search space of molecules. Provided as a list of paths to ".smi" files
        num_workers: Number of simulation tasks to run in parallel
        inference_chunk_size: Number of molecules to run inference on per task
    """
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

    solution: SingleFidelityActiveLearning

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 solution: SingleFidelityActiveLearning,
                 search_space: list[Path | str],
                 database: MoleculeStore,
                 num_workers: int = 2,
                 inference_chunk_size: int = 10000):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, recipes, solution, search_space, database)
        self.search_space_dir = self.run_dir / 'search-space'
        self.scorer = solution.scorer
        self._cache_search_space(inference_chunk_size, search_space)

        # Startup-related information
        self.starter = self.solution.starter

        # Model tracking information
        self.models = solution.models.copy()
        if len(set(map(len, self.models))) > 1:  # pragma: no-coverage
            raise ValueError('You must provide the same number of models for each class')
        if len(self.models) != len(recipes):  # pragma: no-coverage
            raise ValueError('You must provide as many model ensembles as recipes')
        self._model_proxy_keys: list[list[ConnectorKeyT | None]] = [[None] * len(m) for m in self.models]  # Proxy for the current model(s)
        self._ready_models: Queue[tuple[int, int]] = Queue()  # Queue of models are ready for inference

        # Partition the search space into smaller chunks
        self.search_space_smiles: list[list[str]]
        self.search_space_inputs: list[list[object]]
        self.search_space_smiles, self.search_space_inputs = zip(*self._cache_search_space(inference_chunk_size, self.search_space))

        # Coordination tools
        self.start_inference: Event = Event()
        self.start_training: Event = Event()

    @property
    def num_models(self) -> int:
        """Number of models being trained by this class"""
        return sum(map(len, self.models))

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

    def _get_training_set(self, recipe: PropertyRecipe) -> list[MoleculeRecord]:
        """Gather molecules for which the target property is available

        Args:
            recipe: Recipe to evaluate
        Returns:
            List of molecules for which that property is defined
        """
        return [x for x in self.database.iterate_over_records() if recipe.lookup(x) is not None]

    @agent(startup=True)
    def startup(self):
        """Pre-populate the database, if needed."""

        # Determine how many training points are available
        train_size = min(len(self._get_training_set(recipe)) for recipe in self.recipes)

        # If enough, start by training
        if train_size > self.solution.minimum_training_size:
            self.logger.info(f'Training set is larger than the threshold size ({train_size}>{self.solution.minimum_training_size}). Starting model training')
            self.start_training.set()
            return

        # If not, pick some
        self.logger.info(f'Training set is smaller than the threshold size ({train_size}<{self.solution.minimum_training_size})')
        subset = self.starter.select(list(interleave_longest(*self.search_space_smiles)), self.num_to_run)
        self.logger.info(f'Selected {len(subset)} molecules to run')
        with self.task_queue_lock:
            for key in subset:
                self.task_queue.append((key, np.nan))  # All get the same score
            self.task_queue_lock.notify_all()

    @event_responder(event_name='start_training')
    def retrain(self):
        """Retrain all models"""

        # Check if training is still ongoing
        if self.start_inference.is_set():
            self.logger.info('Inference is still ongoing. Will not retrain yet')
            return

        for recipe_id, recipe in enumerate(self.recipes):
            # Get the training set
            train_set = self._get_training_set(recipe)
            self.logger.info(f'Gathered a total of {len(train_set)} entries for retraining recipe {recipe_id}')

            # If too small, stop
            if len(train_set) < self.solution.minimum_training_size:
                self.logger.info(f'Too few to entries to train. Waiting for {self.solution.minimum_training_size}')
                return

            # Process to form the inputs and outputs
            train_inputs = self.scorer.transform_inputs(train_set)
            train_outputs = self.scorer.transform_outputs(train_set, recipe)
            self.logger.info('Pre-processed the training entries')

            # Submit all models
            for model_id, model in enumerate(self.models[recipe_id]):
                model_msg = self.scorer.prepare_message(model, training=True)
                self.queues.send_inputs(
                    model_msg, train_inputs, train_outputs,
                    method='retrain',
                    topic='train',
                    task_info={'recipe_id': recipe_id, 'model_id': model_id}
                )
            self.logger.info(f'Submitted all models for recipe={recipe_id}')

        # Retrieve the results
        for i in range(self.num_models):
            result = self.queues.get_result(topic='train')
            self._write_result(result, 'train')
            assert result.success, f'Training failed: {result.failure_info}'

            # Update the appropriate model
            model_id = result.task_info['model_id']
            recipe_id = result.task_info['recipe_id']
            model_msg = result.value
            if isinstance(model_msg, Proxy):
                # Forces resolution. Needed to avoid `submit_inference` from making a proxy of `model_msg`, which can happen if it is not resolved
                #  by `scorer.update` and is a problem because the proxy for `model_msg` can be evicted while other processes need it
                model_msg = extract(model_msg)
            self.models[recipe_id][model_id] = self.scorer.update(self.models[recipe_id][model_id], model_msg)
            self.logger.info(f'Updated model {i + 1}/{self.num_models}. Recipe id={recipe_id}. Model id={model_id}')

            # Signal to begin inference
            self.start_inference.set()
            self._ready_models.put((recipe_id, model_id))
        self.logger.info('Finished training all models')

    def submit_inference(self) -> tuple[np.ndarray, list[np.ndarray]]:
        """Submit all molecules to be evaluated, return placeholders for their outputs

        Returns:
            - Boolean array marking if inference task are done ``n_chunks x recipes x ensemble_size x ``
            - List of arrays in which to store inference results a total of ``n_chunks`` arrays of size ``recipes x batch_size x models ``
        """

        # Get the proxystore for inference, if defined
        store = self.inference_store

        # Submit a model as soon as it is read
        for i in range(self.num_models):
            # Wait for a model to finish training
            recipe_id, model_id = self._ready_models.get()
            model = self.models[recipe_id][model_id]

            # Serialize and, if available, proxy the model
            model_msg = self.scorer.prepare_message(model, training=False)
            if store is not None:
                model_msg = store.proxy(model_msg)
                self._model_proxy_keys[recipe_id][model_id] = get_key(model_msg)
            self.logger.info(f'Preparing to submit tasks for model {i + 1}/{self.num_models}.')

            for chunk_id, (chunk_inputs, chunk_keys) in enumerate(zip(self.search_space_inputs, self.search_space_smiles)):
                self.queues.send_inputs(
                    model_msg, chunk_inputs,
                    method='score',
                    topic='inference',
                    task_info={'recipe_id': recipe_id, 'model_id': model_id, 'chunk_id': chunk_id, 'chunk_size': len(chunk_keys)}
                )
            self.logger.info(f'Submitted all tasks for recipe={recipe_id} model={model_id}')

        # Prepare to store the inference results
        n_chunks = len(self.search_space_inputs)
        ensemble_size = len(self.models[0])
        all_done: np.ndarray = np.zeros((n_chunks, len(self.recipes), ensemble_size), dtype=bool)
        inference_results: list[np.ndarray] = [
            np.zeros((len(self.recipes), len(chunk), ensemble_size)) for chunk in self.search_space_smiles
        ]  # (chunk, recipe, molecule, model)
        return all_done, inference_results

    @event_responder(event_name='start_inference')
    def run_inference(self):
        """Store inference results then update the task list"""

        # Submit the tasks and prepare the storage
        all_done, inference_results = self.submit_inference()

        # Reset the selector
        selector = self.solution.selector
        selector.update(self.database, self.recipes)
        selector.start_gathering()

        # Gather all inference results
        n_tasks = len(all_done[0]) * len(all_done)
        self.logger.info(f'Prepared to receive {n_tasks} results')
        for i in range(n_tasks):
            # Find which result this is
            result = self.queues.get_result(topic='inference')
            start_time = perf_counter()
            recipe_id = result.task_info['recipe_id']
            model_id = result.task_info['model_id']
            chunk_id = result.task_info['chunk_id']
            self.logger.info(f'Received inference result {i + 1}/{n_tasks}. Recipe={recipe_id}, model={model_id}, chunk={chunk_id}, success={result.success}')

            # Save the outcome
            self._write_result(result, 'inference')
            assert result.success, f'Inference failed due to {result.failure_info}'

            # Update the inference results
            all_done[chunk_id, recipe_id, model_id] = True
            inference_results[chunk_id][recipe_id, :, model_id] = np.squeeze(result.value)

            # Check if we are done for the whole chunk (all models for this chunk)
            if all_done[:, :, chunk_id].all():
                self.logger.info(f'Everything done for chunk={chunk_id}. Adding to selector.')
                selector.add_possibilities(self.search_space_smiles[chunk_id], inference_results[chunk_id])

            # If we are done with the model
            if all_done[:, model_id, :].all():
                self.logger.info(f'Done with all inference tasks for model={model_id}. Evicting.')
                if self._model_proxy_keys[recipe_id][model_id] is not None:
                    self.inference_store.evict(self._model_proxy_keys[recipe_id][model_id])

            # Mark that we're done with this result
            self.logger.info(f'Done processing inference result {i + 1}/{n_tasks}. Time: {perf_counter() - start_time:.2e}s')

        # Get the top list of molecules
        self.logger.info('Done storing all results')
        with self.task_queue_lock:
            self.task_queue.clear()
            for key, score in selector.dispense():
                self.task_queue.append((str(key), score))

            # Notify anyone waiting on more tasks
            self.task_queue_lock.notify_all()
        self.logger.info('Updated task queue. All done.')

    def _simulations_complete(self, record: MoleculeRecord):
        self.start_training.set()
