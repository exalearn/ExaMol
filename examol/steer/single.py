"""Single-objective and single-fidelity implementation of active learning. As easy as we get"""
from pathlib import Path
from threading import Event, Condition
from typing import Iterable, Iterator

import numpy as np
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import event_responder, task_submitter, ResourceCounter, result_processor

from .base import MoleculeThinker
from ..score.base import Scorer
from ..select.base import Selector
from ..simulate.initialize import generate_inchi_and_xyz
from ..store.models import MoleculeRecord
from ..store.recipes import RedoxEnergy


class SingleObjectiveThinker(MoleculeThinker):
    """A thinker which submits all computations needed to evaluate a molecule whenever it is selected"""

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipe: RedoxEnergy,
                 database: list[MoleculeRecord],
                 models: list[Scorer],
                 selector: Selector,
                 num_to_run: int,
                 search_space: Iterable[tuple[str, object]],
                 num_workers: int = 2,
                 inference_chunk_size: int = 10000):
        """

        Args:
            queues: Queues used to communicate with the task server
            run_dir: Directory in which to store logs, etc.
            recipe: Recipe used to compute the target property
            database: List of molecules which are already known
            models: Models used to predict target property
            selector: Tool used to pick which computations to run
            num_to_run: Number of molecules to evaluate
            search_space: Search space of molecules. Provided as an iterator over pairs of SMILES string and molecule in format ready for use with models
            num_workers: Number of simulation tasks to run in parallel
            inference_chunk_size: Number of molecules to run inference on per task
        """
        super().__init__(queues, ResourceCounter(num_workers), run_dir, search_space, inference_chunk_size)

        # Store the selection equipment
        self.database: dict[str, MoleculeRecord] = dict((record.key, record) for record in database)
        self.models = models.copy()
        self.selector = selector

        # Attributes related to simulation
        self.recipe = recipe
        self.task_queue_lock: Condition = Condition()
        self.task_queue: list[tuple[str, float]] = []  # List of tasks to run, SMILES string and score
        self.task_iterator = self._task_iterator()  # Tool for pulling from the task queue

        # Track progress
        self.num_to_run: int = num_to_run
        self.completed: int = 0

        # Coordination tools
        self.start_inference: Event = Event()
        self.start_training: Event = Event()

        # Start by training
        self.start_training.set()

    def _task_iterator(self) -> Iterator[tuple[MoleculeRecord, str, int, str]]:
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

            # Have it run a relaxation task for both neutral and charged
            #  TODO (wardlt): Let the recipe tell me what to do
            _, xyz = generate_inchi_and_xyz(smiles)
            yield record, xyz, 0, None
            yield record, xyz, self.recipe.charge, None

    @task_submitter()
    def submit_simulation(self):
        """Submit a simulation task when resources are available"""
        record, xyz, charge, solvent = next(self.task_iterator)
        self.logger.info(f'Optimizing structure for {record.key} with a charge of {charge}')
        self.queues.send_inputs(
            xyz, self.recipe.energy_config, charge, solvent,
            method='optimize_structure',
            topic='simulation',
            task_info={'key': record.key, 'config_name': self.recipe.energy_config, 'charge': charge}
        )

    @result_processor(topic='simulation')
    def store_simulation(self, result: Result):
        """Store the output of a simulation"""
        assert result.method == 'optimize_structure', 'We only support optimization tasks for now'

        # Trigger a new simulation to start
        self.rec.release()

        # Get the molecule record
        mol_key = result.task_info["key"]
        record = self.database[mol_key]
        self.logger.info(f'Received a result for {mol_key}. Runtime={result.time_running:.1f}s, success={result.success}')

        # Add our result, see if finished
        if result.success:
            sim_result, steps, metadata = result.value
            record.add_energies(sim_result, steps)

            value = self.recipe.lookup(record, recompute=True)
            if value is not None:
                self.completed += 1
                if self.completed == self.num_to_run:
                    self.done.set()
                self.logger.info(f'Finished computing recipe for {mol_key}')
                self.start_training.set()

        self._write_result(result, 'simulation')

    @event_responder(event_name='start_inference')
    def submit_inference(self):
        """Submit all molecules to be evaluated"""

        # Loop over models first to improve caching (avoid-reloading model if needed)
        for model_id, model in enumerate(self.models):
            model_msg = model.get_model_state()
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
                self.task_queue.append((key, score))

            # Notify anyone waiting on more tasks
            self.task_queue_lock.notify_all()
        self.logger.info('Updated task queue. All done.')

    @event_responder(event_name='start_training')
    def retrain(self):
        """Retrain all models"""

        # Get the training set
        train_set = [x for x in list(self.database.values()) if self.recipe.lookup(x) is not None]
        self.logger.info(f'Gathered a total of {len(train_set)} entries for retraining')

        # Process to form the inputs and outputs
        train_inputs = self.models[0].transform_inputs(train_set)
        train_outputs = self.models[0].transform_outputs(train_set)
        self.logger.info('Pre-processed the training entries')

        # Submit all models
        for model_id, model in enumerate(self.models):
            model_msg = model.get_model_state()
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
            self.models[model_id].update(model_msg)
            self.logger.info(f'Updated model {i + 1}/{len(self.models)}. Model id={model_id}')
        self.logger.info('Finished training all models')

        self.start_inference.set()
