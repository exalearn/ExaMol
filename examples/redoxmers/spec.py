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
from examol.steer.single import SingleStepThinker
from examol.store.recipes import RedoxEnergy
from examol.select.baseline import GreedySelector
from examol.specify import ExaMolSpecification

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

# Mark how we report outcomes
reporter = MarkdownReporter()
writer = DatabaseWriter()

# Make the parsl (compute) and proxystore (optional data fabric) configuration
is_mac = sys.platform == 'darwin'
config = Config(
    executors=[HighThroughputExecutor(max_workers=4, cpu_affinity='none' if is_mac else 'block', address='127.0.0.1')],
    run_dir=str((my_path / 'parsl-logs')),
)
store = Store(name='file', connector=FileConnector(store_dir=str(my_path / 'proxystore')), metrics=True)

spec = ExaMolSpecification(
    database=(my_path / 'training-data.json'),
    recipes=[recipe],
    search_space=[(my_path / 'search_space.smi')],
    selector=GreedySelector(8, maximize=True),
    simulator=ASESimulator(scratch_dir=(my_path / 'tmp')),
    scorer=scorer,
    models=[[pipeline]],
    num_to_run=4,
    thinker=SingleStepThinker,
    compute_config=config,
    proxystore=store,
    reporters=[reporter, writer],
    run_dir=run_dir,
)
