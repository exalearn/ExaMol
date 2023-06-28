"""Tool for defining then deploying an ExaMol application"""
import json
import logging
from csv import reader
from dataclasses import dataclass, field
from functools import update_wrapper
from pathlib import Path
from typing import Iterator

from colmena.queue import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.task_server.base import BaseTaskServer
from more_itertools import batched
from parsl import Config
from pydantic.main import partial

from examol.reporting.base import BaseReporter
from examol.score.base import Scorer
from examol.select.base import Selector
from examol.simulate.base import BaseSimulator
from examol.start.base import Starter
from examol.start.fast import RandomStarter
from examol.steer.base import MoleculeThinker
from examol.steer.single import SingleObjectiveThinker
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe

logger = logging.getLogger(__name__)


@dataclass
class ExaMolSpecification:
    """Specification for a molecular design application that can set up then start it.

    **Creating a Compute Configuration**

    The :attr:`compute_config` option accepts a subset of Parsl's configuration options:

    - *Single Executor*: Specify a single executor and have ExaMol use that executor for all tasks
    - *Split Executor*: Specify two executors and label one "learning" and the other "simulation"
      to have the AI tasks be placed on one resource and simulation on the other.
    """

    # Define the problem
    database: Path | str = ...
    """Path to the initial dataset"""
    recipe: PropertyRecipe = ...
    """Definition for how to compute the target property"""
    search_space: Path | str = ...
    """Path to the molecules over which to search. Can either be a `.smi` file or a `.csv` where the first column
    is the smiles string and the second is a form ready for inference with :attr:`scorer`."""
    starter: Starter = RandomStarter(threshold=10, min_to_select=1)
    """How to initialize the database if too small. Default: Pick a single random molecule"""
    selector: Selector = ...
    """How to identify which computation to perform next"""
    scorer: Scorer = ...
    """Defines algorithms used to retrain and run :attr:`models`"""
    models: list[object] = ...
    """List of machine learning models used to predict outcome of :attr:`recipe`"""
    simulator: BaseSimulator = ...
    """Tool used to perform quantum chemistry computations"""
    num_to_run: int = ...
    """Number of quantum chemistry computations to perform"""

    # Options for key operations
    train_options: dict = field(default_factory=dict)
    """Options passed to the :py:meth:`~examol.score.base.Scorer.retrain` function"""

    # Define how we create the thinker
    thinker: type[SingleObjectiveThinker] = ...
    """Policy used to schedule computations"""
    thinker_options: dict[str, object] = field(default_factory=dict)

    # Define how we communicate to the user
    reporters: list[BaseReporter] = field(default_factory=list)
    """List of classes which provide users with real-time information"""

    # Define the computing resources
    compute_config: Config = ...
    """Description of the available resources via Parsl. See :class:`~parsl.config.Config`."""
    run_dir: Path | str = ...
    """Path in which to write output files"""

    def assemble(self) -> tuple[BaseTaskServer, MoleculeThinker]:
        """Assemble the Colmena application"""

        # Use pipe queues for simplicity
        # TODO (wardlt): Add proxystore support
        queues = PipeQueues(topics=['inference', 'simulation', 'train'])

        # Pin options to some functions
        def _wrap_function(fun, options: dict):
            wrapped_fun = partial(fun, **options)
            update_wrapper(wrapped_fun, fun)
            return wrapped_fun

        train_func = _wrap_function(self.scorer.retrain, self.train_options)

        # Determine how methods are partitioned to executors
        exec_names = set(x.label for x in self.compute_config.executors)
        if len(exec_names) == 1:  # Case 1: All to the
            methods = [self.scorer.score, train_func, self.simulator.optimize_structure, self.simulator.compute_energy]
        elif exec_names == {'learning', 'simulation'}:
            methods = [(x, {'executors': ['learning']}) for x in [self.scorer.score, self.scorer.retrain]]
            methods += [(x, {'executors': ['simulation']}) for x in [self.simulator.optimize_structure, self.simulator.compute_energy]]
        else:
            raise NotImplementedError(f'We do not support the executor layout: {",".join(exec_names)}')

        # Create the doer
        doer = ParslTaskServer(
            queues=queues,
            methods=methods,
            config=self.compute_config,
        )

        # Create the thinker
        thinker = self.thinker(
            queues=queues,
            run_dir=self.run_dir,
            recipe=self.recipe,
            search_space=self.load_search_space(),
            starter=self.starter,
            database=self.load_database(),
            scorer=self.scorer,
            models=self.models.copy(),
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
        if self.database.exists():
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
