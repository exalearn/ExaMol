"""Specification of the optimization problem"""
from pathlib import Path

from parsl import Config, HighThroughputExecutor
from parsl.addresses import address_by_hostname
from parsl.launchers import SrunLauncher
from parsl.providers import SlurmProvider

from examol.reporting.database import DatabaseWriter
from examol.reporting.markdown import MarkdownReporter
from examol.score.rdkit import make_knn_model, RDKitScorer
from examol.simulate.ase import ASESimulator
from examol.start.fast import RandomStarter
from examol.steer.single import SingleObjectiveThinker
from examol.store.recipes import RedoxEnergy
from examol.select.baseline import GreedySelector
from examol.specify import ExaMolSpecification

# Get my path. We'll want to provide everything as absolute paths, as they are relative to this file
my_path = Path().absolute()

# Make the recipe
recipe = RedoxEnergy(1, energy_config='gaussian_b3lyp_6-31(2df,p)', solvent='acn')

# Make the scorer
pipeline = make_knn_model()
scorer = RDKitScorer()

# Mark how we report outcomes
reporter = MarkdownReporter()
writer = DatabaseWriter()

# Define how to run Gaussian
sim = ASESimulator(
    scratch_dir=(my_path / 'tmp'),
    ase_db_path='data.db',
    gaussian_command='g16',
)

# Make the parsl configuration
htex = HighThroughputExecutor(
    address=address_by_hostname(),
    max_workers=1,  # Only one task per job
    provider=SlurmProvider(
        partition='knlall',
        launcher=SrunLauncher(),
        nodes_per_block=1,
        max_blocks=1,
        scheduler_options="#SBATCH --account=ML-for-Redox",
        worker_init='''
module load gaussian/16-a.03
export GAUSS_SCRDIR=/scratch
export GAUSS_WDEF="$(scontrol show hostname $SLURM_JOB_NODELIST | paste -d, -s)"
export GAUSS_CDEF=0-63
export GAUSS_MDEF=30GB
export GAUSS_SDEF=ssh
export GAUSS_LFLAGS="-vv"''',
        walltime="20:00:00"
    )
)
config = Config(
    executors=[htex],
    run_dir=str((my_path / 'parsl-logs')),
)

# Build the specification
run_dir = my_path / 'run'
spec = ExaMolSpecification(
    database=run_dir / 'database.json',
    recipe=recipe,
    search_space=(my_path / 'search_space.smi'),
    selector=GreedySelector(8, maximize=True),
    starter=RandomStarter(threshold=10, min_to_select=1),
    simulator=sim,
    scorer=scorer,
    models=[pipeline] * 8,
    num_to_run=20,
    thinker=SingleObjectiveThinker,
    compute_config=config,
    reporters=[reporter, writer],
    run_dir=run_dir,
)
