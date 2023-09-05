"""Compute an initial dataset for various recipes"""
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from threading import Event
from queue import LifoQueue
from pathlib import Path
from shutil import move
import logging
import sys

from colmena.models import Result
from rdkit import RDLogger
from proxystore.connectors.file import FileConnector
from proxystore.store import Store, register_store
from colmena.thinker import BaseThinker, ResourceCounter, event_responder, task_submitter, result_processor
from colmena.task_server.parsl import ParslTaskServer
from colmena.queue import PipeQueues

from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy

import configs

RDLogger.DisableLog('rdApp.*')


# Make the steering application
class BruteForceThinker(BaseThinker):
    """Evaluate all molecules in a user-provided list

    Args:
        queues: Queues used to communicate with task server
        args: Arguments provided to the run script
        slots: Number of molecules to run concurrently
        database: All known information about each molecule
        recipes: List of recipes to be computed
        database_path: Path in which to store the database
        optimization_path: Path in which to store optimization steps
    """

    def __init__(self,
                 queues: PipeQueues,
                 args: Namespace,
                 slots: int,
                 database: dict[str, MoleculeRecord],
                 recipes: list[RedoxEnergy],
                 database_path: Path,
                 optimization_path: Path | None):
        super().__init__(queues, ResourceCounter(slots))
        self.optimization_path = optimization_path
        self.database_path = database_path
        self.database = database
        self.recipes = recipes
        self.args = args

        # Determine where to store the task records
        run_name = self.database_path.name[:-5]
        self.record_path = self.database_path.parent / f'{run_name}-results.json'

        # State tracking
        self.molecule_queue: LifoQueue[str] = LifoQueue()  # List of molecules which are yet to be computed
        self.ongoing_tasks: dict[str, int] = defaultdict(int)  # How many tasks ongoing for each molecule
        self.checkpoint = Event()

        # Populate the list of molecules to run
        for key, _ in zip(database.keys(), range(len(self.database) if args.num_to_run is None else args.num_to_run)):
            self.molecule_queue.put(key)

    @task_submitter()
    def submit_task(self):
        """Submit tasks for a new molecule"""

        # Loop until we either find new work or run out of molecules
        while not self.molecule_queue.empty():
            # Get the next record
            key = self.molecule_queue.get()
            my_record = self.database[key]

            # See if there is any new work to do
            for recipe in self.recipes:
                try:
                    next_calculations = recipe.suggest_computations(my_record)
                    if len(next_calculations) > 0:
                        my_logger.debug(f'Submitting tasks for {my_record.key} recipe {recipe.name}@{recipe.level}')
                        # Submit them
                        for request in next_calculations:
                            method = 'optimize_structure' if request.optimize else 'compute_energy'
                            self.queues.send_inputs(
                                my_record.key, request.xyz, request.config_name, request.charge, request.solvent,
                                method=method,
                                task_info={'key': my_record.key, 'recipe': recipe.name, 'level': recipe.level}
                            )
                            self.ongoing_tasks[my_record.key] += 1

                        # Finished!
                        return

                    # Compute the property
                    recipe.update_record(my_record)
                except ValueError as e:
                    logger.warning(f'{my_record.key} failed for {recipe.name}@{recipe.level}. Error: {e}')
                    if self.args.halt_on_error:
                        raise ValueError(f'Failed to submit new tasks for {my_record.key}')

        # If there are neither molecules no ongoing tasks, then we are done
        if len(self.ongoing_tasks) == 0:
            self.done.set()

    @result_processor()
    def store_result(self, result: Result):
        """Store the result from a computation"""

        # Get the associated record
        my_record = self.database[result.task_info['key']]
        self.ongoing_tasks[my_record.key] -= 1

        # Save the result to disk
        with self.record_path.open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)

        # Update the record
        if not result.success:
            self.logger.warning(f'Computation failed for {my_record.key}. Error: {result.failure_info}')
            if args.halt_on_error:
                raise ValueError(f'Computation failed. Halting everything')
        else:
            if result.method == 'optimize_structure':
                sim_result, steps, metadata = result.value
                my_record.add_energies(sim_result, steps)

                # Save the steps to disk
                if self.optimization_path is not None:
                    with self.optimization_path.open('a') as fp:
                        for step in steps:
                            print(step.json(), file=fp)
            else:
                sim_result, metadata = result.value
                my_record.add_energies(sim_result)

        # Check if we need more work for this molecule
        self.logger.debug(f'Stored record for {my_record.key}. {self.ongoing_tasks[my_record.key]} tasks remaining')
        if self.ongoing_tasks[my_record.key] == 0:
            self.molecule_queue.put(my_record.key)
            self.ongoing_tasks.pop(my_record.key)
            self.rec.release()

        # Trigger the engine to checkpoint our database
        self.checkpoint.set()

    @event_responder(event_name='checkpoint')
    def write_database(self):
        """Save the database to disk"""
        temp_path = self.database_path.parent / f'{self.database_path.name}.new'
        with temp_path.open('w') as fp:
            for record in dataset.values():
                print(record.to_json(), file=fp)
        move(temp_path, self.database_path)


if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('search_space', help='Path to the SMI file containing strings to be run')
    parser.add_argument('--config-function', default='make_local_config', help='Name of the configuration function from `config.py`')
    parser.add_argument('--num-to-run', default=None, help='Maximum number of molecules to run', type=int)
    parser.add_argument('--halt-on-error', action='store_true', help='Halt the workflow if a single task fails')
    parser.add_argument('--write-frequency', default=5, help='Minimum frequency of saving database', type=float)
    args = parser.parse_args()

    # Make a logger
    my_logger = logging.getLogger('main')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    for logger in [logging.getLogger('examol'), my_logger]:
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    # Load in the molecules
    search_path = Path(args.search_space)
    my_logger.info(f'Loading molecules from {search_path}')
    with search_path.open() as fp:
        molecules = [line.strip() for line in fp]
    my_logger.info(f'Loaded {len(molecules)} molecules to screen')

    # Get the path to the dataset
    dataset_path = Path('datasets') / f'{search_path.name[:-4]}.json'
    records_path = dataset_path.parent / f'{search_path.name[:-4]}-simulation-records.json'
    dataset_path.parent.mkdir(exist_ok=True)
    dataset = {}

    # Make the database
    if dataset_path.is_file():
        # Load the existing data
        my_logger.info(f'Loading initial data from {dataset_path}')
        with dataset_path.open() as fp:
            for line in fp:
                record = MoleculeRecord.from_json(line)
                dataset[record.key] = record

        assert len(dataset) == len(molecules), f'There was a corrupted save. Dataset has {len(dataset)} records, expected {len(molecules)}'
    else:
        my_logger.info(f'Creating an initial dataset at {dataset_path}')
        # Ensure each molecule in the search space is in this database
        with dataset_path.open('w') as fp:
            for smiles in molecules:
                record = MoleculeRecord.from_identifier(smiles)
                if record.key not in dataset:
                    dataset[record.key] = record
                print(record.to_json(), file=fp)
    my_logger.info(f'Starting from a dataset of {len(dataset)} records')

    # Get the right computational environment
    config_fn = getattr(configs, args.config_function)
    config, sim, n_slots, energy_configs = config_fn()
    my_logger.info(f'Loaded configuration function "{args.config_function}"')

    # Get the recipes we should run
    recipes = []
    for energy_level in energy_configs:
        for charge in [-1, 1]:
            for solvent in [None, 'acn']:
                recipes.extend([
                    RedoxEnergy(energy_config=energy_level, vertical=True, charge=charge, solvent=solvent),
                    RedoxEnergy(energy_config=energy_level, vertical=False, charge=charge, solvent=solvent),
                ])
    my_logger.info(f'Assembled a list of {len(recipes)} recipes to compute')

    # Create the queues which will connect task server and thinker
    store = Store(name='file', connector=FileConnector(store_dir='proxy-store'))
    register_store(store)
    queues = PipeQueues(proxystore_name='file', proxystore_threshold=10000)

    # Create the task server
    task_server = ParslTaskServer(queues=queues, methods=[sim.compute_energy, sim.optimize_structure], config=config)
    my_logger.info('Created task server')

    # Create the thinker
    thinker = BruteForceThinker(queues, args, n_slots, dataset, recipes, dataset_path, records_path)
    thinker.logger.addHandler(handler)
    thinker.logger.setLevel(logging.INFO)

    # Run the script
    try:
        task_server.start()
        thinker.run()
    finally:
        queues.send_kill_signal()

    task_server.join()

    # Save the result one last time
    temp_path = dataset_path.parent / f'{dataset_path.name}.new'
    with temp_path.open('w') as fp:
        for record in dataset.values():
            print(record.to_json(), file=fp)
    move(temp_path, dataset_path)
    my_logger.info(f'Wrote database to {dataset_path}')

    # Close the proxystore
    store.close()
    my_logger.info('Done!')
