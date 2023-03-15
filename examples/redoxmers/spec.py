"""Specification of the optimization problem"""
from parsl import Config, HighThroughputExecutor

from examol.score.rdkit import make_knn_model, RDKitScorer
from examol.simulate.ase import ASESimulator
from examol.steer.single import SingleObjectiveThinker
from examol.store.recipes import RedoxEnergy
from examol.select.baseline import GreedySelector
from examol.specify import ExaMolSpecification
from pathlib import Path

# Get my path. We'll want to provide everything as absolute paths, as they are relative to this file
my_path = Path()

# Make the recipe
recipe = RedoxEnergy(1, energy_config='xtb')

# Make the scorer
pipeline = make_knn_model()
scorer = RDKitScorer(recipe)

# Make the parsl configuration
config = Config(
    executors=[HighThroughputExecutor(max_workers=4, cpu_affinity='block')],
    run_dir=str((my_path / 'parsl-logs').absolute()),
)

spec = ExaMolSpecification(
    database=(my_path / 'training-data.json').absolute(),
    recipe=recipe,
    search_space=(my_path / 'search_space.smi').absolute(),
    selector=GreedySelector(8, maximize=True),
    simulator=ASESimulator(scratch_dir=(my_path / 'tmp').absolute()),
    scorer=scorer,
    models=[pipeline],
    num_to_run=8,
    thinker=SingleObjectiveThinker,
    compute_config=config,
    run_dir=(my_path / 'run').absolute()
)
