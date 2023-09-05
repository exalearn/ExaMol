"""Compute an initial dataset for various recipes"""
from concurrent.futures import Future, wait, ALL_COMPLETED
from argparse import ArgumentParser
from pathlib import Path
import logging
import sys

from rdkit import RDLogger
from parsl.app.python import PythonApp
import parsl

from examol.simulate.base import SimResult
from examol.store.models import MoleculeRecord
from examol.store.recipes import RedoxEnergy

import configs

RDLogger.DisableLog('rdApp.*')

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

    #  Get the right computational environment
    config_fn = getattr(configs, args.config_function)
    config, sim, energy_configs = config_fn()
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

    # Start Parsl and wrap the workflow functions
    parsl.load(config)
    energy_app = PythonApp(sim.compute_energy)
    relax_app = PythonApp(sim.optimize_structure)
    my_logger.info(f'Loaded Parsl and build applications')

    # Submit an initial batch of work
    ongoing_tasks: dict[str, int] = dict((x, 0) for x in dataset)  # How many tasks ongoing for each molecule
    futures: list[Future] = []


    def submit_new_work(my_record: MoleculeRecord) -> list[Future]:
        """Submit all the new tasks for a record

        Args:
            my_record: Record to be evaluated

        Returns:
            The futures produced by the calculations,
            with the attribute ".key" of each set as the key for the record
            and the attribute of ".optimize" set to whether this is an optimization or not
        """

        for recipe in recipes:
            try:
                next_calculations = recipe.suggest_computations(my_record)
                if len(next_calculations) > 0:
                    my_logger.debug(f'Submitting tasks for {my_record.key} recipe {recipe.name}@{recipe.level}')
                    # Submit them
                    my_futures = []
                    for request in next_calculations:
                        app = relax_app if request.optimize else energy_app
                        my_future = app(my_record.key, request.xyz, request.config_name, request.charge, request.solvent)
                        my_future.key = my_record.key
                        my_future.optimize = request.optimize
                        my_futures.append(my_future)

                    # Update the ongoing task list
                    ongoing_tasks[my_record.key] += len(my_futures)
                    return my_futures

                # Compute the property
                recipe.update_record(my_record)
            except ValueError as e:
                logger.warning(f'{my_record.key} failed for {recipe.name}@{recipe.level}. Error: {e}')
                if args.halt_on_error:
                    raise
                return []

        my_logger.debug(f'Done with {len(recipes)} recipes for {my_record.key}')
        return []


    for record, _ in zip(dataset.values(), range(len(dataset) if args.num_to_run is None else args.num_to_run)):
        new_futures = submit_new_work(record)
        futures.extend(new_futures)

    my_logger.info(f'Started an initial batch of {len(futures)} calculations')

    # Store and submit new records as they complete
    while len(futures) > 0:
        # Wait until all completed or 5s
        done, futures = wait(futures, timeout=args.write_frequency, return_when=ALL_COMPLETED)

        # Process completed records
        if len(done) == 0:
            continue
        new_futures = []
        simulation_records: list[SimResult] = []
        my_logger.info(f'{len(done)} tasks have completed')
        for future in done:
            # Get the associated record
            record = dataset[future.key]

            # Update the record
            try:
                result = future.result()
            except BaseException as e:
                my_logger.warning(f'Computation failed for {record.key}. Error: {e}')
                if args.halt_on_error:
                    raise
                continue
            if future.optimize:
                sim_result, steps, metadata = result
                simulation_records.extend([sim_result] + steps)
                record.add_energies(sim_result, steps)
            else:
                sim_result, metadata = result
                simulation_records.append(sim_result)
                record.add_energies(sim_result)

            # Check if we need more work for this molecule
            ongoing_tasks[record.key] -= 1
            my_logger.debug(f'Stored record for {record.key}. {ongoing_tasks[record.key]} tasks remaining')
            if ongoing_tasks[record.key] == 0:
                my_logger.debug(f'Submitting new work for {record.key}')
                new_futures.extend(submit_new_work(record))

        futures.update(new_futures)
        my_logger.info(f'Submitted {len(new_futures)} calculations. Total ongoing tasks is {len(futures)}')

        # Save the database
        with dataset_path.open('w') as fp:
            for record in dataset.values():
                print(record.to_json(), file=fp)
        my_logger.info(f'Wrote updated dataset to {dataset_path}')

        # Write the energy/forces to disk
        #  TODO (wardlt): Write to a temporary path and then copy
        with records_path.open('a') as fp:
            for record in simulation_records:
                print(record.json(), file=fp)
        my_logger.info(f'Appended {len(simulation_records)} new simulation records to {records_path}')
