"""Single-objective and single-fidelity implementation of active learning. As easy as we get"""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from queue import Queue
from threading import Event, Condition
from time import perf_counter
from typing import Iterator, Sequence

import numpy as np
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import event_responder, task_submitter, ResourceCounter, result_processor, agent
from more_itertools import interleave_longest
from proxystore.proxy import Proxy, extract
from proxystore.store.base import ConnectorKeyT
from proxystore.store.utils import get_key

from .base import MoleculeThinker
from ..score.base import Scorer
from ..select.base import Selector
from ..simulate.base import SimResult
from ..start.base import Starter
from ..store.models import MoleculeRecord
from ..store.recipes import PropertyRecipe, SimulationRequest


class SingleStepThinker(MoleculeThinker):
    """A thinker which submits all computations needed to evaluate a molecule whenever it is selected

    Args:
        queues: Queues used to communicate with the task server
        run_dir: Directory in which to store logs, etc.
        recipes: Recipes used to compute the target properties
        database: List of molecules which are already known
        scorer: Tool used as part of model training
        models: Models used to predict target property. We require the same ensemble size for each recipe
        selector: Tool used to pick which computations to run
        starter: How to pick calculations before enough data are available
        num_to_run: Number of molecules to evaluate
        search_space: Search space of molecules. Provided as a list of paths to ".smi" files
        num_workers: Number of simulation tasks to run in parallel
        inference_chunk_size: Number of molecules to run inference on per task
    """

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 database: list[MoleculeRecord],
                 scorer: Scorer,
                 models: list[list[object]],
                 starter: Starter,
                 selector: Selector,
                 num_to_run: int,
                 search_space: list[Path | str],
                 num_workers: int = 2,
                 inference_chunk_size: int = 10000):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, search_space, scorer, database, inference_chunk_size)

        # Store the selection equipment
        if len(set(map(len, models))) > 1:  # pragma: no-coverage
            raise ValueError('You must provide the same number of models for each class')
        if len(models) != len(recipes):  # pragma: no-coverage
            raise ValueError('You must provide as many model ensembles as recipes')
        self.models = models.copy()
        self.selector = selector
        self.starter = starter

        # Attributes related to simulation
        self.recipes = tuple(recipes)
        self.task_queue_lock: Condition = Condition()
        self.task_queue: list[tuple[str, float]] = []  # List of tasks to run, SMILES string and score
        self.task_iterator = self._task_iterator()  # Tool for pulling from the task queue

        # Track progress
        self.num_to_run: int = num_to_run
        self.completed: int = 0
        self.molecules_in_progress: dict[str, int] = defaultdict(int)  # Map of InChI Key -> number of ongoing computations

        # Model tracking information
        self._model_proxy_keys: list[list[ConnectorKeyT | None]] = [[None] * len(m) for m in self.models]  # Proxy for the current model
        self._ready_models: Queue[tuple[int, int]] = Queue()  # Queue of models are ready for inference

        # Coordination tools
        self.start_inference: Event = Event()
        self.start_training: Event = Event()

    @property
    def num_models(self) -> int:
        """Number of models being trained by this class"""
        return sum(map(len, self.models))

    def _task_iterator(self) -> Iterator[tuple[MoleculeRecord, SimulationRequest]]:
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
                self.database[record.key] = record

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

    @agent(startup=True)
    def startup(self):
        """Pre-populate the database, if needed."""

        # Determine how many training points are available
        train_size = min(len(self._get_training_set(recipe)) for recipe in self.recipes)

        # If enough, start by training
        if train_size > self.starter.threshold:
            self.logger.info(f'Training set is larger than the threshold size ({train_size}>{self.starter.threshold}). Starting model training')
            self.start_training.set()
            return

        # If not, pick some
        self.logger.info(f'Training set is smaller than the threshold size ({train_size}<{self.starter.threshold})')
        subset = self.starter.select(list(interleave_longest(*self.search_space_smiles)), self.num_to_run)
        self.logger.info(f'Selected {len(subset)} molecules to run')
        with self.task_queue_lock:
            for key in subset:
                self.task_queue.append((key, np.nan))  # All get the same score
            self.task_queue_lock.notify_all()

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
            not_done = sum(recipe.lookup(record, recompute=True) is None for recipe in self.recipes)
            if not_done == 0:
                # If so, mark that we have finished computing the property
                self.completed += 1
                self.logger.info(f'Finished computing all recipes for {mol_key}. Completed {self.completed}/{self.num_to_run} molecules')
                if self.completed == self.num_to_run:
                    self.logger.info('Done!')
                    self.done.set()
                self.start_training.set()

                # Mark that we've finished with all recipes
                result.task_info['status'] = 'finished'
                result.task_info['result'] = [recipe.lookup(record) for recipe in self.recipes]
            else:
                # If not, see if we need to resubmit to finish the computation
                self.logger.info(f'Finished {len(self.recipes) - not_done}/{len(self.recipes)} recipes for {mol_key}')
                result.task_info['status'] = 'in progress'
                if self.molecules_in_progress[mol_key] == 0:
                    self.logger.info('We must submit new computations for this molecule. Re-adding it to the front of the task queue')
                    with self.task_queue_lock:
                        self.task_queue.insert(0, (record.identifier.smiles, np.inf))
                        self.task_queue_lock.notify_all()

            # Save the relaxation steps to disk
            with open(self.run_dir / 'simulation-records.json', 'a') as fp:
                for record in results:
                    print(record.json(), file=fp)

        self._write_result(result, 'simulation')

    @event_responder(event_name='start_inference')
    def submit_inference(self):
        """Submit all molecules to be evaluated"""

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

    @event_responder(event_name='start_inference')
    def store_inference(self):
        """Store inference results then update the task list"""
        # Prepare to store the inference results
        n_chunks = len(self.search_space_inputs)
        ensemble_size = len(self.models[0])
        all_done: np.ndarray = np.zeros((len(self.recipes), ensemble_size, n_chunks), dtype=bool)  # Whether a chunk is finished. (recipe, chunk, model)
        inference_results: list[np.ndarray] = [
            np.zeros((len(self.recipes), len(chunk), ensemble_size)) for chunk in self.search_space_smiles
        ]  # (chunk, recipe, molecule, model)
        n_tasks = self.num_models * n_chunks

        # Reset the selector
        self.selector.update(self.database, self.recipes)
        self.selector.start_gathering()

        # Gather all inference results
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
            all_done[recipe_id, model_id, chunk_id] = True
            inference_results[chunk_id][recipe_id, :, model_id] = np.squeeze(result.value)

            # Check if we are done for the whole chunk (all models for this chunk)
            if all_done[:, :, chunk_id].all():
                self.logger.info(f'Everything done for chunk={chunk_id}. Adding to selector.')
                self.selector.add_possibilities(self.search_space_smiles[chunk_id], inference_results[chunk_id])

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
            for key, score in self.selector.dispense():
                self.task_queue.append((str(key), score))

            # Notify anyone waiting on more tasks
            self.task_queue_lock.notify_all()
        self.logger.info('Updated task queue. All done.')

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
            if len(train_set) < self.starter.threshold:
                self.logger.info(f'Too few to entries to train. Waiting for {self.starter.threshold}')
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

    def _get_training_set(self, recipe: PropertyRecipe) -> list[MoleculeRecord]:
        """Gather molecules for which the target property is available

        Args:
            recipe: Recipe to evaluate
        Returns:
            List of molecules for which that property is defined
        """
        return [x for x in list(self.database.values()) if recipe.lookup(x) is not None]
