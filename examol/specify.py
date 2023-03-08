"""Tool for defining then deploying an ExaMol application"""
import json
import logging
from csv import reader
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from colmena.queue import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.task_server.base import BaseTaskServer
from more_itertools import batched
from parsl import Config

from examol.score.base import Scorer
from examol.select.base import Selector
from examol.simulate.base import BaseSimulator
from examol.steer.base import MoleculeThinker
from examol.steer.single import SingleObjectiveThinker
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe

logger = logging.getLogger(__name__)


@dataclass
class ExaMolSpecification:
    """Specification for a molecular design application that can set up then start it

    Attributes:
        simulator: Tool used to perform quantum chemistry computations
        recipe: Definition for how to compute the target property
        selector: How to identify which computation to perform next
        database: Path to the initial dataset
        search_space: Path to the molecules over which to search. Can either be a `.smi` file or a `.csv` where the first column
            is the smiles string and the second is a form ready for inference with :attr:`scorer`.
        thinker: Tool used to schedule computations
        compute_config: Description of the available resources via Parsl. See :class:`~parsl.config.Config`.
        num_to_run: Number of quantum chemistry computations to perform
    """

    # Define the problem
    database: Path | str = ...
    recipe: PropertyRecipe = ...
    search_space: Path | str = ...
    selector: Selector = ...
    scorer: Scorer = ...
    simulator: BaseSimulator = ...
    num_to_run: int = ...

    # Define how we create the thinker
    thinker: type[SingleObjectiveThinker] = ...
    thinker_options: dict[str, object] = field(default_factory=dict)

    # Define the computing resources
    compute_config: Config = ...
    run_dir: Path | str = ...

    def assemble(self) -> tuple[BaseTaskServer, MoleculeThinker]:
        """Assemble the Colmena application"""

        queues = PipeQueues(topics=['inference', 'simulation', 'train'])
        doer = ParslTaskServer(
            queues=queues,
            methods=[self.scorer.score, self.simulator.optimize_structure, self.scorer.retrain],
            config=self.compute_config,
        )

        thinker = self.thinker(
            queues=queues,
            run_dir=self.run_dir,
            recipe=self.recipe,
            search_space=self.load_search_space(),
            database=self.load_database(),
            models=[self.scorer],
            selector=self.selector,
            num_to_run=self.num_to_run,
            **self.thinker_options
        )

        return doer, thinker

    def load_database(self) -> list[MoleculeRecord]:
        """Load the starting database

        Returns:
            List of molecules defined
        """

        output = []
        with open(self.database) as fp:
            for line in fp:
                output.append(MoleculeRecord.from_json(line))
        logger.info(f'Loaded {len(output)} records from {self.database}')
        return output

    def load_search_space(self) -> Iterator[tuple[str, object]]:
        """Load the search space incrementally"""

        path = Path(self.search_space)
        if path.name.lower().endswith('.smi'):
            # Read then convert to inputs
            with path.open() as fp:
                # Generate the inputs in batches
                for batch in batched(fp, 1024):  # TODO (wardlt): This could be configurable and parallelized
                    batch = [MoleculeRecord.from_identifier(x.strip()) for x in batch]
                    yield from zip(
                        [record.identifier.smiles for record in batch],
                        self.scorer.transform_inputs(batch)
                    )
        elif path.name.lower().endswith('.csv'):
            with path.open() as fp:
                csv = reader(fp)
                for row in csv:
                    yield row[0], json.loads(row[1])
        else:
            raise ValueError(f'File type is unrecognized for {path}')
