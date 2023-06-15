"""Single-objective and single-fidelity implementation of active learning. As easy as we get"""
from threading import Event, Condition
from typing import Iterable, Iterator
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import event_responder, task_submitter, ResourceCounter, result_processor, agent
from more_itertools import interleave_longest

from .base import MoleculeThinker
from ..score.base import Scorer
from ..select.base import Selector
from ..start.base import Starter
from ..store.models import MoleculeRecord
from ..store.recipes import PropertyRecipe, SimulationRequest


class SingleObjectiveThinker(MoleculeThinker):
    """A thinker which submits all computations needed to evaluate a molecule whenever it is selected

    Args:
        queues: Queues used to communicate with the task server
        run_dir: Directory in which to store logs, etc.
        recipe: Recipe used to compute the target property
        database: List of molecules which are already known
        scorer: Tool used as part of model training
        models: Models used to predict target property
        selector: Tool used to pick which computations to run
        starter: How to pick calculations before enough data are available
        num_to_run: Number of molecules to evaluate
        search_space: Search space of molecules. Provided as an iterator over pairs of SMILES string and molecule in format ready for use with models
        num_workers: Number of simulation tasks to run in parallel
        inference_chunk_size: Number of molecules to run inference on per task
    """

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipe: PropertyRecipe,
                 database: list[MoleculeRecord],
                 scorer: Scorer,
                 models: list[object],
                 starter: Starter,
                 selector: Selector,
                 num_to_run: int,
                 search_space: Iterable[tuple[str, object]],
                 num_workers: int = 2,
                 inference_chunk_size: int = 10000):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, search_space, database, inference_chunk_size)

        # Store the selection equipment
        self.models = models.copy()
        self.scorer = scorer
        self.selector = selector
        self.starter = starter

        # Attributes related to simulation
        self.recipe = recipe
        self.task_queue_lock: Condition = Condition()
        self.task_queue: list[tuple[str, float]] = []  # List of tasks to run, SMILES string and score
        self.task_iterator = self._task_iterator()  # Tool for pulling from the task queue

        # Track progress
        self.num_to_run: int = num_to_run
        self.completed: int = 0
        self.molecules_in_progress: dict[str, int] = defaultdict(int)  # Map of InChI Key -> number of ongoing computations

        # Coordination tools
        self.start_inference: Event = Event()
        self.start_training: Event = Event()

    def _task_iterator(self) -> Iterator[tuple[MoleculeRecord, SimulationRequest]]:
        """Iterate over the next tasks in the task queue"""

        while True:
            # Get the next molecule to run
            with self.task_queue_lock:
                if len(self.task_queue) == 0:
                    self.logger.info('No tasks available to run. Waiting')
                    self.task_queue_lock.wait()
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
                suggestions = self.recipe.suggest_computations(record)
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
        train_size = len(self._get_training_set())

        # If enough, start by training
        if train_size > self.starter.threshold:
            self.logger.info(f'Training set is larger than the threshold size ({train_size}>{self.starter.threshold}). Starting model training')
            self.start_training.set()
            return

        # If not, pick some
        self.logger.info(f'Training set is smaller than the threshold size ({train_size}<{self.starter.threshold})')
        needed = self.starter.threshold - train_size
        subset = self.starter.select(interleave_longest(*self.search_space_keys), needed)
        self.logger.info(f'Selected {len(subset)} molecules to run')
        with self.task_queue_lock:
            for key in subset:
                self.task_queue.append((key, np.nan))  # All get the same score

    @task_submitter()
    def submit_simulation(self):
        """Submit a simulation task when resources are available"""
        record, suggestion = next(self.task_iterator)

        if suggestion.optimize:
            self.logger.info(f'Optimizing structure for {record.key} with a charge of {suggestion.charge}')
            self.queues.send_inputs(
                suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent,
                method='optimize_structure',
                topic='simulation',
                task_info={'key': record.key, **asdict(suggestion)}
            )
        else:
            self.logger.info(f'Getting single-point energy for {record.key} with a charge of {suggestion.charge} ' +
                             ('' if suggestion.solvent is None else f'in {suggestion.solvent}'))
            self.queues.send_inputs(
                suggestion.xyz, suggestion.config_name, suggestion.charge, suggestion.solvent,
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
        self.logger.info(f'Received a result for {mol_key}. Runtime={result.time_running:.1f}s, success={result.success}')

        # Update the number of computations in progress
        self.molecules_in_progress[mol_key] -= 1

        # Add our result, see if finished
        if result.success:
            if result.method == 'optimize_structure':
                sim_result, steps, metadata = result.value
                record.add_energies(sim_result, steps)
            elif result.method == 'compute_energy':
                sim_result, metadata = result.value
                record.add_energies(sim_result)
            else:
                raise NotImplementedError()

            # If we can compute then property than we are done
            value = self.recipe.lookup(record, recompute=True)
            if value is not None:
                # If so, mark that we have finished computing the property
                self.completed += 1
                if self.completed == self.num_to_run:
                    self.done.set()
                self.logger.info(f'Finished computing recipe for {mol_key}. Completed {self.completed}/{self.num_to_run} molecules')
                self.start_training.set()

                # Mark that we've finished with this recipe
                result.task_info['status'] = 'finished'
                result.task_info['result'] = value
            else:
                # If not, see if we need to resubmit to finish the computation
                result.task_info['status'] = 'in progress'
                if self.molecules_in_progress[mol_key] == 0:
                    self.logger.info('We must submit new computations for this molecule. Re-adding it to the front of the task queue')
                    with self.task_queue_lock:
                        self.task_queue.insert(0, (record.identifier.smiles, np.inf))
                        self.task_queue_lock.notify_all()

        self._write_result(result, 'simulation')

    @event_responder(event_name='start_inference')
    def submit_inference(self):
        """Submit all molecules to be evaluated"""

        # Loop over models first to improve caching (avoid-reloading model if needed)
        for model_id, model in enumerate(self.models):
            model_msg = self.scorer.prepare_message(model, training=False)
            for chunk_id, (chunk_inputs, chunk_keys) in enumerate(zip(self.search_space_inputs, self.search_space_keys)):
                self.queues.send_inputs(
                    model_msg, chunk_inputs,
                    method='score',
                    topic='inference',
                    task_info={'model_id': model_id, 'chunk_id': chunk_id, 'chunk_size': len(chunk_keys)}
                )
            self.logger.info(f'Submitted all tasks for {model_id}')

    @event_responder(event_name='start_inference')
    def store_inference(self):
        """Store inference results then update the task list"""
        # Prepare to store the inference results
        n_models, n_chunks = len(self.models), len(self.search_space_inputs)
        all_done: list[list[bool]] = [[False] * n_models for _ in range(n_chunks)]  # Whether a certain chunk has finished. (chunk, model)
        inference_results: list[np.ndarray] = [np.zeros((len(chunk), n_models)) for chunk in self.search_space_keys]  # Values for each chunk
        n_tasks = n_models * n_chunks

        # Reset the selector
        self.selector.update(self.database, self.recipe)
        self.selector.start_gathering()

        # Gather all inference results
        self.logger.info(f'Prepared to receive {n_tasks} results')
        for i in range(n_tasks):
            # Find which result this is
            result = self.queues.get_result(topic='inference')
            model_id = result.task_info['model_id']
            chunk_id = result.task_info['chunk_id']
            self.logger.info(f'Received inference result {i + 1}/{n_tasks}. Model={model_id}, chunk={chunk_id}, success={result.success}')

            # Save the outcome
            self._write_result(result, 'inference')
            assert result.success, f'Inference failed due to {result.failure_info}'

            # Update the inference results
            all_done[chunk_id][model_id] = True
            inference_results[chunk_id][:, model_id] = result.value

            # Check if we are done for the whole chunk (all models for this chunk)
            if all(all_done[chunk_id]):
                self.logger.info(f'Everything done for chunk={chunk_id}')
                self.selector.add_possibilities(self.search_space_keys[chunk_id], inference_results[chunk_id])

            # Mark that we're done with this result
            self.logger.info(f'Done processing inference result {i + 1}/{n_tasks}')

        # Get the top list of molecules
        self.logger.info('Done storing all results')
        self.selector.start_dispensing()
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

        # Get the training set
        train_set = self._get_training_set()
        self.logger.info(f'Gathered a total of {len(train_set)} entries for retraining')

        # If too small, stop
        if len(train_set) < self.starter.threshold:
            self.logger.info(f'Too few to entries to train. Waiting for {self.starter.threshold}')
            return

        # Process to form the inputs and outputs
        train_inputs = self.scorer.transform_inputs(train_set)
        train_outputs = self.scorer.transform_outputs(train_set, self.recipe)
        self.logger.info('Pre-processed the training entries')

        # Submit all models
        for model_id, model in enumerate(self.models):
            model_msg = self.scorer.prepare_message(model, training=True)
            self.queues.send_inputs(
                model_msg, train_inputs, train_outputs,
                method='retrain',
                topic='train',
                task_info={'model_id': model_id}
            )
        self.logger.info('Submitted all models')

        # Gather the results
        for i in range(len(self.models)):
            result = self.queues.get_result(topic='train')
            self._write_result(result, 'train')
            assert result.success, f'Training failed: {result.failure_info}'

            # Update the appropriate model
            model_id = result.task_info['model_id']
            model_msg = result.value
            self.models[model_id] = self.scorer.update(self.models[model_id], model_msg)
            self.logger.info(f'Updated model {i + 1}/{len(self.models)}. Model id={model_id}')
        self.logger.info('Finished training all models')

        self.start_inference.set()

    def _get_training_set(self) -> list[MoleculeRecord]:
        """Gather molecules for which the target property is available"""
        return [x for x in list(self.database.values()) if self.recipe.lookup(x) is not None]
