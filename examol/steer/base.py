"""Base class that defines core routines used across many steering policies"""
import gzip
import json
import logging
from pathlib import Path
from typing import Iterator

from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, ResourceCounter

from examol.store.models import MoleculeRecord


class MoleculeThinker(BaseThinker):
    """Base for a thinker which performs molecular design

    Args:
        queues: Queues used to communicate with the task server
        rec: Counter used to track availability of different resources
        run_dir: Directory in which to store results
        database: List of molecule records
        search_space: Lists of molecules to be evaluated as a list of ".smi" or ".sdf" files
    """

    database: dict[str, MoleculeRecord]
    """Map of InChI key to molecule record"""

    def __init__(self,
                 queues: ColmenaQueues,
                 rec: ResourceCounter,
                 run_dir: Path,
                 search_space: list[Path | str],
                 database: list[MoleculeRecord]):
        super().__init__(queues, resource_counter=rec)
        self.database: dict[str, MoleculeRecord] = dict((record.key, record) for record in database)
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.search_space = search_space

        # Mark where the logs should be stored
        handler = logging.FileHandler(self.run_dir / 'run.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        for logger in [self.logger, logging.getLogger('colmena'), logging.getLogger('proxystore')]:
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

    def iterate_over_search_space(self, only_smiles: bool = False) -> Iterator[MoleculeRecord]:
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
                            yield MoleculeRecord.from_json(line)
                        else:
                            yield MoleculeRecord.from_identifier(line.strip())
            else:
                raise ValueError(f'File type is unrecognized for {path}')

    def _write_result(self, result: Result, result_type: str):
        with (self.run_dir / f'{result_type}-results.json').open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)
