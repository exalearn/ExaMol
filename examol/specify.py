"""Tool for defining then deploying an ExaMol application"""
from dataclasses import dataclass, field
from functools import update_wrapper
from functools import partial
from typing import Sequence
from pathlib import Path
import logging

from colmena.queue import PipeQueues
from colmena.task_server import ParslTaskServer
from colmena.task_server.base import BaseTaskServer
from parsl import Config
from proxystore.store import Store, register_store

from examol.reporting.base import BaseReporter
from examol.score.base import Scorer
from examol.select.base import Selector
from examol.simulate.base import BaseSimulator
from examol.start.base import Starter
from examol.start.fast import RandomStarter
from examol.steer.base import MoleculeThinker
from examol.steer.single import SingleStepThinker
from examol.store.models import MoleculeRecord
from examol.store.recipes.base import PropertyRecipe

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
    recipes: Sequence[PropertyRecipe] = ...
    """Definition for how to compute the target properties"""
    search_space: list[Path | str] = ...
    """Path to the molecules over which to search. Should be a list of ".smi" files"""
    starter: Starter = RandomStarter(threshold=10)
    """How to initialize the database if too small. Default: Pick a single random molecule"""
    selector: Selector = ...
    """How to identify which computation to perform next"""
    scorer: Scorer = ...  # TODO (wardlt): Support a different type of model for each recipe
    """Defines algorithms used to retrain and run :attr:`models`"""
    models: list[list[object]] = ...
    """List of machine learning models used to predict outcome of :attr:`recipes`"""
    simulator: BaseSimulator = ...
    """Tool used to perform quantum chemistry computations"""
    num_to_run: int = ...
    """Number of quantum chemistry computations to perform"""

    # Options for key operations
    train_options: dict = field(default_factory=dict)
    """Options passed to the :py:meth:`~examol.score.base.Scorer.retrain` function"""
    score_options: dict = field(default_factory=dict)
    """Options passed to the :py:meth:`~examol.score.base.Scorer.score` function"""

    # Define how we create the thinker
    thinker: type[SingleStepThinker] = ...
    """Policy used to schedule computations"""
    thinker_options: dict[str, object] = field(default_factory=dict)
    """Options passed forward to initializing the thinker"""

    # Define how we communicate to the user
    reporters: list[BaseReporter] = field(default_factory=list)
    """List of classes which provide users with real-time information"""

    # Define the computing resources
    compute_config: Config = ...
    """Description of the available resources via Parsl. See :class:`~parsl.config.Config`."""
    proxystore: Store | dict[str, Store] | None = None
    """Proxy store(s) used to communicate large objects between Thinker and workers. Can be either a single store used for all task types,
    or a mapping between a task topic (inference, simulation, train) and the store used for that task type.

    All messages larger than 10kB will be proxied using the store."""
    run_dir: Path | str = ...
    """Path in which to write output files"""

    def assemble(self) -> tuple[BaseTaskServer, MoleculeThinker]:
        """Assemble the Colmena application"""

        # Use pipe queues for simplicity
        if self.proxystore is None:
            proxy_name = None
        elif isinstance(self.proxystore, Store):
            register_store(self.proxystore, exist_ok=True)
            proxy_name = self.proxystore.name
            logger.info(f'Will use {self.proxystore} for all messages')
        elif isinstance(self.proxystore, dict):
            proxy_name = dict()
            for name, store in self.proxystore.items():
                register_store(store, exist_ok=True)
                proxy_name[name] = store.name
                logger.info(f'Using {store} for {name} tasks')
        else:
            raise NotImplementedError()
        queues = PipeQueues(topics=['inference', 'simulation', 'train'], proxystore_threshold=10000, proxystore_name=proxy_name)

        # Pin options to some functions
        def _wrap_function(fun, options: dict):
            wrapped_fun = partial(fun, **options)
            update_wrapper(wrapped_fun, fun)
            return wrapped_fun

        train_func = _wrap_function(self.scorer.retrain, self.train_options)
        score_func = _wrap_function(self.scorer.score, self.score_options)

        # Determine how methods are partitioned to executors
        exec_names = set(x.label for x in self.compute_config.executors)
        if len(exec_names) == 1:  # Case 1: All to on the same executor
            methods = [score_func, train_func, self.simulator.optimize_structure, self.simulator.compute_energy]
        elif exec_names == {'learning', 'simulation'}:  # Case 2: Split ML and simulation
            methods = [(x, {'executors': ['learning']}) for x in [score_func, train_func]]
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
            recipes=self.recipes,
            search_space=self.search_space,
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
        logger.info(f'Loading records from {self.database}')
        if self.database.exists():
            with open(self.database) as fp:
                for line in fp:
                    output.append(MoleculeRecord.from_json(line))
            logger.info(f'Loaded {len(output)} molecule property records')
        return output
