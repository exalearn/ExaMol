"""Specification of the optimization problem"""
from functools import partial
from pathlib import Path

from colmena.queue.redis import RedisQueues
from parsl import Config, HighThroughputExecutor
from parsl.addresses import address_by_hostname
from proxystore.connectors.file import FileConnector
from proxystore.store import Store

from examol.reporting.markdown import MarkdownReporter
from examol.score.rdkit import RDKitScorer, make_knn_model, FingerprintTransformer
from examol.simulate.ase import ASESimulator
from examol.solution import MultiFidelityActiveLearning
from examol.start.fast import RandomStarter
from examol.steer.multifi import PipelineThinker
from examol.store.recipes import RedoxEnergy, SolvationEnergy
from examol.select.botorch import EHVISelector
from examol.specify import ExaMolSpecification

# Get my path. We'll want to provide everything as absolute paths, as they are relative to this file
my_path = Path().absolute()
num_workers: int = 4
multi_fidelity: bool = False
random: bool = True
colmena_queues = partial(RedisQueues)  # Using a partial in case we need to pin options, such as port numbers

# Make a run directory named for the settings
run_dir = my_path / (('run' if multi_fidelity else 'run-single') + ('-random' if random else ''))

# Make the recipes to be optimized
recipes = [
    RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn'),
    SolvationEnergy('mopac_pm7', 'acn')
]

# Make the scorer
pipeline = make_knn_model(n_neighbors=4)
transform: FingerprintTransformer = pipeline.steps.pop(0)[1]
transform.n_jobs = 1
pipeline[0].n_jobs = 1
scorer = RDKitScorer(pre_transform=transform)

# Mark how we report outcomes
reporter = MarkdownReporter()

# Define how to run Gaussian
sim = ASESimulator(
    scratch_dir=(my_path / 'ase-runs'),
)

# Make the workflow configuration
store = Store(name='file', connector=FileConnector(str(run_dir / 'proxystore')), metrics=True)
htex = HighThroughputExecutor(
    address=address_by_hostname(),
    max_workers=num_workers,
    cpu_affinity='block'
)
config = Config(
    executors=[htex],
    run_dir=str((my_path / 'parsl-logs')),
)


# Mark that we're going to solve this with multifidelity learning
num_to_run = 200
solution = MultiFidelityActiveLearning(
    selector=EHVISelector(200, maximize=[True, False]),
    steps=[  # The order of codes to run
        [RedoxEnergy(1, energy_config='mopac_pm7', vertical=True)],
        [RedoxEnergy(1, energy_config='mopac_pm7', vertical=True, solvent='acn')]
    ] if multi_fidelity else [],
    starter=RandomStarter(),
    minimum_training_size=num_to_run if random else 10,
    scorer=scorer,
    models=[[pipeline] * 8] * 2,  # 8 models for 2 properties
    num_to_run=num_to_run,
    pipeline_target=0.5,
)

# Build the specification
spec = ExaMolSpecification(
    database=run_dir / 'database.json',
    recipes=recipes,
    solution=solution,
    search_space=[(my_path / '..' / 'redoxmers-bebop' / 'search-space.smi')],
    simulator=sim,
    thinker=PipelineThinker,
    compute_config=config,
    reporters=[reporter],
    run_dir=run_dir,
    proxystore=store,
    proxystore_threshold=10000,  # Needs to be small to avoid filling Colmena's PipeQueues
    colmena_queue=colmena_queues,
    thinker_options={'num_workers': num_workers}
)
