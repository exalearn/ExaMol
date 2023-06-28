"""Specification of the optimization problem"""
from pathlib import Path
import shutil

from parsl import Config, HighThroughputExecutor

from examol.reporting.database import DatabaseWriter
from examol.reporting.markdown import MarkdownReporter
from examol.score.rdkit import make_knn_model, RDKitScorer
from examol.simulate.ase import ASESimulator
from examol.steer.single import SingleObjectiveThinker
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
recipe = RedoxEnergy(1, energy_config='xtb', solvent='acn')

# Make the scorer
pipeline = make_knn_model()
scorer = RDKitScorer()

# Mark how we report outcomes
reporter = MarkdownReporter()
writer = DatabaseWriter()

# Make the parsl configuration
config = Config(
    executors=[HighThroughputExecutor(max_workers=4, cpu_affinity='block', address='localhost')],
    run_dir=str((my_path / 'parsl-logs')),
)

spec = ExaMolSpecification(
    database=(my_path / 'training-data.json'),
    recipe=recipe,
    search_space=[(my_path / 'search_space.smi')],
    selector=GreedySelector(8, maximize=True),
    simulator=ASESimulator(scratch_dir=(my_path / 'tmp')),
    scorer=scorer,
    models=[pipeline],
    num_to_run=4,
    thinker=SingleObjectiveThinker,
    compute_config=config,
    reporters=[reporter, writer],
    run_dir=run_dir,
)
