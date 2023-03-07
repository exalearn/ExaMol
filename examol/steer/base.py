"""Base class that defines core routines used across many steering policies"""
import logging
from pathlib import Path
from typing import Iterable

from more_itertools import batched
from colmena.models import Result
from colmena.queue import ColmenaQueues
from colmena.thinker import BaseThinker, ResourceCounter


class MoleculeThinker(BaseThinker):
    """Base for a thinker which performs molecular design

    Attributes:
        run_dir: Directory in which to store outputs
        search_space_keys: Keys associated with each molecule in the search space, broken into chunks
        search_space_inputs: Inputs to the ML models for each molecule in the search space, broken into chucks
    """

    def __init__(self,
                 queues: ColmenaQueues,
                 rec: ResourceCounter,
                 run_dir: Path,
                 search_space: Iterable[tuple[str, object]],
                 inference_chunk_size: int = 10000):
        """
        Args:
            queues: Queues used to communicate with the task server
            rec: Resource used to track tasks on different resources
            run_dir: Directory in which to store results
        """
        super().__init__(queues, resource_counter=rec)
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Mark where the logs should be stored
        handler = logging.FileHandler(self.run_dir / 'run.log')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

        # Partition the search space into smaller chunks
        self.search_space_keys: list[list[str]] = []
        self.search_space_inputs: list[list[object]] = []
        for batch in batched(search_space, inference_chunk_size):
            batch_keys, batch_inputs = zip(*batch)
            self.search_space_keys.append(batch_keys)
            self.search_space_inputs.append(batch_inputs)

    def _write_result(self, result: Result, result_type: str):
        with (self.run_dir / f'{result_type}-results.json').open('a') as fp:
            print(result.json(exclude={'value', 'inputs'}), file=fp)
