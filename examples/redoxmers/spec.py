"""Specification of the optimization problem"""
from pathlib import Path
import shutil
import sys

from parsl import Config, HighThroughputExecutor
from proxystore.store import Store
from proxystore.connectors.file import FileConnector

from examol.reporting.database import DatabaseWriter
from examol.reporting.markdown import MarkdownReporter
from examol.score.rdkit import make_knn_model, RDKitScorer
from examol.simulate.ase import ASESimulator
from examol.specify.solution import SingleFidelityActiveLearning
from examol.start.fast import RandomStarter
from examol.steer.single import SingleStepThinker
from examol.store.recipes import RedoxEnergy
from examol.select.baseline import GreedySelector
from examol.specify import ExaMolSpecification

# Parameters you may want to configure
num_random: int = 2  # Number of randomly-selected molecules to run
num_total: int = 8  # Total number of molecules to run

# Get my path. We'll want to provide everything as absolute paths, as they are relative to this file
my_path = Path().absolute()

# Delete the old run
run_dir = my_path / 'run'
if run_dir.is_dir():
    shutil.rmtree(run_dir)

# Make the recipe
recipe = RedoxEnergy(1, energy_config='mopac_pm7', solvent='acn')

# Make the scorer
pipeline = make_knn_model()
scorer = RDKitScorer()

# Define the tools needed to solve the problem
solution = SingleFidelityActiveLearning(
    starter=RandomStarter(),
    minimum_training_size=num_random,
    selector=GreedySelector(num_total, maximize=True),
    scorer=scorer,
    models=[[pipeline]],
    num_to_run=num_total,
)

# Mark how we report outcomes
reporter = MarkdownReporter()
writer = DatabaseWriter()

# Make the parsl (compute) and proxystore (optional data fabric) configuration
is_mac = sys.platform == 'darwin'
config = Config(
    executors=[HighThroughputExecutor(max_workers=1)],
    run_dir=str((my_path / 'parsl-logs')),
)
store = Store(name='file', connector=FileConnector(store_dir=str(my_path / 'proxystore')), metrics=True)

spec = ExaMolSpecification(
    database=(my_path / 'training-data.json'),
    recipes=[recipe],
    search_space=[(my_path / 'search_space.smi')],
    solution=solution,
    simulator=ASESimulator(scratch_dir=(run_dir / 'tmp'), clean_after_run=False),
    thinker=SingleStepThinker,
    compute_config=config,
    proxystore=store,
    reporters=[reporter, writer],
    run_dir=run_dir,
)
