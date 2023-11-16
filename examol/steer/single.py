"""Single-objective and single-fidelity implementation of active learning. As easy as we get"""
from pathlib import Path
from time import perf_counter
from typing import Sequence

import numpy as np
from colmena.queue import ColmenaQueues
from colmena.thinker import event_responder, ResourceCounter
from proxystore.store.utils import get_key

from .base import ScorerThinker
from examol.solution import SingleFidelityActiveLearning
from ..store.db.base import MoleculeStore
from ..store.models import MoleculeRecord
from ..store.recipes import PropertyRecipe


class SingleStepThinker(ScorerThinker):
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

    def __init__(self,
                 queues: ColmenaQueues,
                 run_dir: Path,
                 recipes: Sequence[PropertyRecipe],
                 database: MoleculeStore,
                 solution: SingleFidelityActiveLearning,
                 search_space: list[Path | str],
                 num_workers: int = 2,
                 inference_chunk_size: int = 10000):
        super().__init__(queues, ResourceCounter(num_workers), run_dir, recipes, solution.scorer, solution, search_space, database, inference_chunk_size)

        # Store the selection equipment
        self.solution = solution
        self.selector = self.solution.selector

    def _simulations_complete(self, record: MoleculeRecord):
        self.start_training.set()

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
