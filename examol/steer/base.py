"""Base class that defines core routines used across many steering policies"""
import gzip
import json
import logging
import os
import pickle as pkl
import shutil
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from more_itertools import batched
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, ResourceCounter

from examol.score.base import Scorer
from examol.store.models import MoleculeRecord


def _generate_inputs(smiles: str, scorer: Scorer) -> tuple[str, object]:
    """Parse a molecule then generate a form ready for inference

    Args:
        smiles: Molecule to be parsed
        scorer: Tool used for inference
    Returns:
        - Key for the molecule record
        - Inference-ready format
        Or None if the transformation fails
    """
    try:
        record = MoleculeRecord.from_identifier(smiles.strip())
        readied = scorer.transform_inputs([record])[0]
    except (ValueError, RuntimeError):
        return None
    return smiles, readied


class MoleculeThinker(BaseThinker):
    """Base for a thinker which performs molecular design

    Args:
        queues: Queues used to communicate with the task server
        rec: Resource used to track tasks on different resources
        run_dir: Directory in which to store results
        search_space: Lists of molecules to be evaluated as a list of ".smi" files
        scorer: Tool which will be used to score the search space
        database: List of molecule records
        inference_chunk_size: How many molecules per inference task

    Attributes:
        search_space_keys: Keys associated with each molecule in the search space, broken into chunks
        search_space_inputs: Inputs to the ML models for each molecule in the search space, broken into chucks
        database: Map between molecule InChI key and currently-known information about it
    """

    def __init__(self,
                 queues: ColmenaQueues,
                 rec: ResourceCounter,
                 run_dir: Path,
                 search_space: list[str | Path],
                 scorer: Scorer,
                 database: list[MoleculeRecord],
                 inference_chunk_size: int = 10000):
        super().__init__(queues, resource_counter=rec)
        self.database: dict[str, MoleculeRecord] = dict((record.key, record) for record in database)
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.scorer = scorer

        # Mark where the logs should be stored
        handler = logging.FileHandler(self.run_dir / 'run.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for logger in [self.logger, logging.getLogger('colmena')]:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

        # Partition the search space into smaller chunks
        self.search_space_dir = self.run_dir / 'search-space'
        self.search_space_keys: list[list[str]]
        self.search_space_inputs: list[list[object]]
        self.search_space_keys, self.search_space_inputs = zip(*self._cache_search_space(inference_chunk_size, search_space))

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
            def _read_molecules():
                """Function to produce a stream of molecules from the input files"""
                for i, path in enumerate(search_space):
                    path = Path(path).resolve()
                    self.logger.info(f'Reading molecules from file {i + 1}/{len(search_space)}: {path.resolve()}')
                    if path.name.lower().endswith('.smi'):
                        with path.open() as fp:
                            for line in fp:
                                yield line.strip()
                    else:
                        raise ValueError(f'File type is unrecognized for {path}')

            # Process the inputs and store them to disk
            search_size = 0
            input_func = partial(_generate_inputs, scorer=self.scorer)
            with ProcessPoolExecutor(min(4, os.cpu_count())) as pool:
                mol_iter = pool.map(input_func, _read_molecules(), chunksize=1000)
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

        # Load in the molecules
        self.logger.info(f'Loading in molecules from {len(search_space_keys)} files')
        output = []
        for name, keys in search_space_keys.items():
            with gzip.open(self.search_space_dir / name, 'rb') as fp:
                output.append((keys, pkl.load(fp)))
        return output

    def _write_result(self, result: Result, result_type: str):
        with (self.run_dir / f'{result_type}-results.json').open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)
