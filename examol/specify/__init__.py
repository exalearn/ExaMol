"""Tool for defining then deploying an ExaMol application"""
import contextlib
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Sequence
from pathlib import Path
import logging

from colmena.queue import PipeQueues, ColmenaQueues
from colmena.task_server import ParslTaskServer
from colmena.task_server.base import BaseTaskServer
from parsl import Config
from proxystore.store import Store, register_store

from examol.reporting.base import BaseReporter
from examol.simulate.base import BaseSimulator
from examol.solution import SolutionSpecification
from examol.steer.base import MoleculeThinker
from examol.store.db.base import MoleculeStore
from examol.store.db.memory import InMemoryStore
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
    database: Path | str | MoleculeStore
    """Path to the data as a line-delimited JSON file or an already-activated store"""
    recipes: Sequence[PropertyRecipe]
    """Definition for how to compute the target properties"""
    search_space: list[Path | str]
    """Path to the molecules over which to search. Should be a list of ".smi" files"""
    simulator: BaseSimulator
    """Tool used to perform quantum chemistry computations"""

    # Define the solution
    solution: SolutionSpecification
    """Define how to solve the design challenge"""

    # Define how we create the thinker
    thinker: type[MoleculeThinker] = ...
    """Policy used to schedule computations"""
    thinker_options: dict[str, object] = field(default_factory=dict)
    """Options passed forward to initializing the thinker"""
    thinker_workers: int = min(4, os.cpu_count())
    """Number of workers to use in the steering process"""

    # Define how we communicate to the user
    reporters: list[BaseReporter] = field(default_factory=list)
    """List of classes which provide users with real-time information"""

    # Define the computing resources
    compute_config: Config = ...
    """Description of the available resources via Parsl. See :class:`~parsl.config.Config`."""
    proxystore: Store | dict[str, Store] | None = None
    """Proxy store(s) used to communicate large objects between Thinker and workers. Can be either a single store used for all task types,
    or a mapping between a task topic (inference, simulation, train) and the store used for that task type.

    All messages larger than :attr:`proxystore_threshold` will be proxied using the store."""
    proxystore_threshold: float | int = 10000
    """Messages larger than this size will be sent via Proxystore rather than through the workflow engine. Units: bytes"""
    colmena_queue: type[ColmenaQueues] = PipeQueues
    """Class used to send messages between Thinker and Task Server."""
    run_dir: Path | str = ...
    """Path in which to write output files"""

    @contextlib.contextmanager
    def assemble(self) -> tuple[BaseTaskServer, MoleculeThinker, MoleculeStore]:
        """Assemble the Colmena application

        Returns:
            - Task server used to perform computations
            - Thinker used to steer computations
            - Store used to collect results
        """

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
        queues = self.colmena_queue(topics=['inference', 'simulation', 'train'], proxystore_threshold=self.proxystore_threshold, proxystore_name=proxy_name)

        # Make the functions associated with steering
        learning_functions = self.solution.generate_functions()

        # Determine how methods are partitioned to executors
        exec_names = set(x.label for x in self.compute_config.executors)
        if len(exec_names) == 1:  # Case 1: All to on the same executor
            methods = learning_functions + [self.simulator.optimize_structure, self.simulator.compute_energy]
        elif exec_names == {'learning', 'simulation'}:  # Case 2: Split ML and simulation
            methods = [(x, {'executors': ['learning']}) for x in learning_functions]
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
        store = self.load_database()
        with store, ProcessPoolExecutor(self.thinker_workers) as pool:
            thinker = self.thinker(
                queues=queues,
                run_dir=self.run_dir,
                recipes=self.recipes,
                search_space=self.search_space,
                solution=self.solution,
                database=store,
                pool=pool,
                **self.thinker_options
            )
            yield doer, thinker, store

    def load_database(self) -> MoleculeStore:
        """Load the starting database

        Returns:
            Pointer to the database object
        """

        if isinstance(self.database, MoleculeStore):
            return self.database
        else:
            return InMemoryStore(self.database)
