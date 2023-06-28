"""Specification of the optimization problem"""
from pathlib import Path
import logging

from parsl.addresses import address_by_hostname
from parsl import Config, HighThroughputExecutor
from parsl.launchers import SimpleLauncher, MpiExecLauncher
from parsl.providers import PBSProProvider
from tensorflow import keras

from examol.reporting.database import DatabaseWriter
from examol.reporting.markdown import MarkdownReporter
from examol.score.nfp import NFPScorer, custom_objects
from examol.select.bayes import ExpectedImprovement
from examol.simulate.ase import ASESimulator
from examol.steer.single import SingleObjectiveThinker
from examol.store.recipes import RedoxEnergy
from examol.specify import ExaMolSpecification

logger = logging.getLogger('spec')

# Often-changed configuration options
num_parallel_cp2k: int = 10
nodes_per_cp2k: int = 2
ensemble_size: int = 8
search_space_name: str = 'pubchem-criteria-v3-molwt=300.smi'

# Get my path. We'll want to provide everything as absolute paths, as they are relative to this file
my_path = Path().absolute()
run_dir = my_path / 'run'

# If we have not run yet, use the database from the chemistry check
database_path = '../0_check-chemistry-settings/database.json'
if run_dir.exists():
    database_path = run_dir / 'database.json'

# Make the recipe
recipe = RedoxEnergy(1, energy_config='cp2k_blyp_dzvp', solvent='acn')

# Make the simulator
sim = ASESimulator(
    scratch_dir='cp2k-files',
    cp2k_command=f'mpiexec -n {nodes_per_cp2k * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                 f'--hostfile /tmp/hostfiles/local_hostfile.`printf %02d $PARSL_WORKER_RANK` '
                 '/lus/grand/projects/CSC249ADCD08/cp2k/set_affinity_gpu_polaris.sh '
                 '/lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp',
)

# Make the scorer
scorer = NFPScorer()
model_path = my_path / '..' / f'1_initial-models/nfp-mpnn/best_models/{recipe.name}/{recipe.level}/model.h5'
model = keras.models.load_model(model_path, custom_objects=custom_objects)

# Mark how we report outcomes
reporter = MarkdownReporter()
writer = DatabaseWriter()

# Make the parsl configuration
config = Config(
    executors=[
        HighThroughputExecutor(  # A single node for all ML tasks
            label='learning',
            cpu_affinity='reverse_block',
            available_accelerators=4,
            provider=PBSProProvider(
                account="CSC249ADCD08",
                worker_init=f"""
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw
module list
cd {my_path}
hostname
pwd

# Load anaconda
conda activate /lus/grand/projects/CSC249ADCD08/ExaMol/env
which python""",
                walltime="6:00:00",
                queue="preemptable",
                scheduler_options="#PBS -l filesystems=home:eagle:grand",
                launcher=MpiExecLauncher(
                    bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                ),
                select_options="ngpus=4",
                nodes_per_block=num_parallel_cp2k * nodes_per_cp2k,
                min_blocks=0,
                max_blocks=1,
                cpus_per_node=64,
            ),
        ),
        HighThroughputExecutor(
            label='simulation',
            address=address_by_hostname(),
            prefetch_capacity=0,
            start_method="fork",  # Needed to avoid interactions between MPI and os.fork
            max_workers=num_parallel_cp2k,
            provider=PBSProProvider(
                account="CSC249ADCD08",
                worker_init=f"""
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw
module list
cd {my_path}
hostname
pwd

# Make the hostfiles for each worker
mkdir -p /tmp/hostfiles/
split --lines={nodes_per_cp2k} --suffix-length=2 $PBS_NODEFILE /tmp/hostfiles/local_hostfile.

# Load anaconda
conda activate /lus/grand/projects/CSC249ADCD08/ExaMol/env
which python
""",
                walltime="6:00:00",
                queue="preemptable",
                scheduler_options="#PBS -l filesystems=home:eagle:grand",
                launcher=SimpleLauncher(),  # Launches only a single copy of the workflows
                select_options="ngpus=4",
                nodes_per_block=num_parallel_cp2k * nodes_per_cp2k,
                min_blocks=0,
                max_blocks=1,
                cpus_per_node=64,
            ),
        ),
    ],
    run_dir=str((my_path / 'parsl-logs')),
)

spec = ExaMolSpecification(
    database=(my_path / 'training-data.json'),
    recipe=recipe,
    search_space=[(my_path / 'search-space/output' / search_space_name)],
    selector=ExpectedImprovement(num_parallel_cp2k * 8, maximize=True),
    simulator=sim,
    scorer=scorer,
    models=[model] * 8,
    num_to_run=32,
    thinker=SingleObjectiveThinker,
    thinker_options={'num_workers': num_parallel_cp2k},
    compute_config=config,
    reporters=[reporter, writer],
    run_dir=run_dir,
)
