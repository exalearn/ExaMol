"""Scheduling strategies for multi-fidelity design campaigns"""
import math
from pathlib import Path
from multiprocessing import Pool
from functools import cached_property
from typing import Sequence, Iterable

import numpy as np
from colmena.queue import ColmenaQueues
from more_itertools import batched

from examol.score.utils.multifi import collect_outputs
from examol.solution import MultiFidelityActiveLearning
from examol.steer.single import SingleStepThinker
from examol.store.db.base import MoleculeStore
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe
from examol.utils.chemistry import get_inchi_key_from_molecule_string


class PipelineThinker(SingleStepThinker):
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
        super().__init__(queues, run_dir, recipes, solution, search_space, database, num_workers, inference_chunk_size)
        self.inference_chunk_size = inference_chunk_size

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
        # Special case: Return first entry if its priority is infinite
        if math.isinf(self.task_queue[0][1]):
            smiles, score = self.task_queue.pop(0)
            record = self.database.get_or_make_record(smiles)
            level = self.get_level(smiles)
            return record, score, self.steps[level]

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
        else:
            self.logger.info(f'Did not find a record at step {target_level}. Opting for one at {current_best[0]}')

        # Return the best choice
        chosen_level, chosen_ind = current_best
        smiles, score = self.task_queue.pop(chosen_ind)
        if chosen_level == -1:
            chosen_level = self.get_level(smiles)
        chosen_level = min(chosen_level, len(self.steps) - 1)  # Defend against race condition where molecule is running last step while being placed in queue
        self.logger.info(f'Pulled molecule at position {chosen_ind} to run at step #{chosen_level}')
        return self.database.get_or_make_record(smiles), score, self.steps[chosen_level]

    def _simulations_complete(self, record: MoleculeRecord):
        super()._simulations_complete(record)
        self.already_in_db.add(record.key)  # Make sure we know it is in the database

        # Put it back on the end of the task queue if we can to run another level
        for recipe in self.recipes:
            if recipe.lookup(record) is None:
                break  # We haven't finished yet
        else:
            self.logger.info(f'Done everything for {record.key}')
            return
        with self.task_queue_lock:
            self.logger.info(f'Still more to do for {record.key}. Putting it back in the queue')
            self.task_queue.append((record.identifier.smiles, 0))

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

    def submit_inference(self) -> tuple[list[list[str]], np.ndarray, list[np.ndarray]]:
        # Submit the tasks from the whole search space
        all_smiles, all_is_done, all_results = super().submit_inference()

        # Submit the tasks from the database
        self.logger.info('Submitting the molecules with data from the database')
        initial_chunks = len(all_smiles)
        store = self.inference_store
        batch_count = 0
        for batch_id, chunk_records in enumerate(
                batched(filter(lambda x: x.key in self.already_in_db, self.database.iterate_over_records()), self.inference_chunk_size)
        ):
            batch_count += 1

            # Prepare the inputs and proxy them, if desired
            all_smiles.append([r.identifier.smiles for r in chunk_records])
            chunk_inputs = self.scorer.transform_inputs(chunk_records)
            if store is not None:
                chunk_inputs = store.proxy(chunk_inputs)

            # Submit models for all chunks
            for recipe_id, recipe in enumerate(self.recipes):
                # Get the lower levels of fidelity
                all_fidelities = self.solution.get_levels_for_property(recipe)
                lower_fidelities = collect_outputs(chunk_records, all_fidelities[:-1])

                for model_id, model in enumerate(self.models[recipe_id]):
                    # Either get the model or the proxy
                    if self._model_proxies[recipe_id][model_id] is None:
                        model_msg = self.scorer.prepare_message(model, training=False)
                    else:
                        model_msg = self._model_proxies[recipe_id][model_id]

                    self.queues.send_inputs(
                        model_msg, chunk_inputs,
                        input_kwargs={'lower_fidelities': lower_fidelities},
                        method='score',
                        topic='inference',
                        task_info={'recipe_id': recipe_id, 'model_id': model_id, 'chunk_id': batch_id + initial_chunks, 'chunk_size': len(chunk_records)}
                    )

            # Create placeholder for the outputs
            all_results.append(np.empty((len(self.recipes), len(chunk_records), len(self.models[0]))))
            self.logger.info(f'Submitted all tasks for batch={batch_id} of records in the database')

        # Append to the "all done" array
        all_is_done = np.concatenate([
            np.zeros((batch_count, len(self.recipes), len(self.models[0])), dtype=bool),
            all_is_done
        ], axis=0)

        return all_smiles, all_is_done, all_results

    def _filter_inference_results(self, chunk_id: int, chunk_smiles: list[str], inference_results: np.ndarray) -> tuple[list[str], np.ndarray]:
        if chunk_id < len(self.search_space_smiles):
            # Remove molecules from the chunk which are in the database
            #  TODO (wardlt): Parallelize this
            mask = [get_inchi_key_from_molecule_string(s) not in self.already_in_db for s in chunk_smiles]
            return [s for m, s in zip(mask, chunk_smiles) if m], inference_results[:, mask, :]
        else:
            return chunk_smiles, inference_results

    def _get_training_set(self, recipe: PropertyRecipe) -> list[MoleculeRecord]:
        # Find all levels at which we compute this property
        to_match_levels = [x.level for x in self.solution.get_levels_for_property(recipe)]
        self.logger.info(f'Finding all records which compute {recipe.name} at levels: {", ".join(to_match_levels)}')

        # Match records that hit _any_ of them
        output = []
        for record in self.database.iterate_over_records():
            if recipe.name in record.properties and any(x in record.properties[recipe.name] for x in to_match_levels):
                output.append(record)
        return output

    def get_additional_training_information(self, train_set: list[MoleculeRecord], recipe: PropertyRecipe) -> dict[str, object]:
        to_match = self.solution.get_levels_for_property(recipe)
        lower_fidelities = collect_outputs(
            train_set, to_match[:-1]
        )
        return dict(lower_fidelities=lower_fidelities)
