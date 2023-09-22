"""Configurations for the different computational environments

Add a new environment by creating a function which produces a Simulation tool,
Parsl config for the desired environment,
the number of execution slots,
and a list of the energy configurations it supports
"""
from parsl.executors import HighThroughputExecutor
from parsl.providers import CobaltProvider, PBSProProvider
from parsl.launchers import AprunLauncher, SimpleLauncher
from parsl.addresses import address_by_interface
from parsl.config import Config

from examol.simulate.ase import ASESimulator


def make_local_config() -> tuple[Config, ASESimulator, int, list[str]]:
    """Make a configuration that will run computations locally"""

    config = Config(
        executors=[HighThroughputExecutor(max_workers=1, address='127.0.0.1')]
    )
    sim = ASESimulator(scratch_dir='run_data')
    return config, sim, 1, ['xtb', 'mopac_pm7']


def make_theta_config() -> tuple[Config, ASESimulator, int, list[str]]:
    """Make a configuration that will run computations on larger theta allocations"""

    max_blocks = 4
    prefetch = 0
    config = Config(
        retries=8,
        executors=[
            HighThroughputExecutor(
                max_workers=1,
                address=address_by_interface('vlan2360'),
                prefetch_capacity=prefetch,
                provider=CobaltProvider(
                    queue='default',
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 256 -j 4 --cc depth"),
                    walltime='0:60:00',
                    nodes_per_block=128,
                    max_blocks=max_blocks,
                    scheduler_options='#COBALT --attrs filesystems=home,theta-fs0:enable_ssh=1',
                    worker_init='''module load miniconda-3; source activate /lus/grand/projects/CSC249ADCD08/ExaMol/env-theta; hostname''',
                    cmd_timeout=120,
                ),
            )]
    )
    sim = ASESimulator(scratch_dir='/tmp/run_data', retain_failed=False)  # Avoid using the global filesystem
    return config, sim, max_blocks * 128 * (prefetch + 1), ['xtb', 'mopac_pm7']


def make_theta_debug_config() -> tuple[Config, ASESimulator, int, list[str]]:
    """Make a configuration that will run computations on a small job"""

    max_blocks = 1
    prefetch = 0
    config = Config(
        retries=8,
        executors=[
            HighThroughputExecutor(
                max_workers=1,
                cpu_affinity='block',
                address=address_by_interface('vlan2360'),
                prefetch_capacity=prefetch,
                provider=CobaltProvider(
                    queue='debug-cache-quad',
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 64 -j 1 --cc depth"),
                    walltime='0:60:00',
                    nodes_per_block=4,
                    max_blocks=max_blocks,
                    scheduler_options='#COBALT --attrs filesystems=home,theta-fs0:enable_ssh=1',
                    worker_init='''module load miniconda-3; source activate /lus/grand/projects/CSC249ADCD08/ExaMol/env-theta; hostname''',
                    cmd_timeout=120,
                ),
            )]
    )
    sim = ASESimulator(scratch_dir='/tmp/run_data', retain_failed=False)
    return config, sim, max_blocks * 4 * (prefetch + 1), ['xtb']


def make_polaris_config() -> tuple[Config, ASESimulator, int, list[str]]:
    """Make a configuration that will run computations on Polaris

    Submits small jobs to the preemptable queue and medium jobs to the prod queue.
    """

    # Settings
    max_blocks = 8
    nodes_per_cp2k = 1
    cp2k_per_block = 2
    prefetch = 0
    nodes_per_block = cp2k_per_block * nodes_per_cp2k

    # Make the config
    config = Config(
        retries=8,
        executors=[
            HighThroughputExecutor(  # Many jobs submitted to the preemptable queue
                max_workers=cp2k_per_block,
                address=address_by_interface('bond0'),
                prefetch_capacity=prefetch,
                provider=PBSProProvider(
                account="CSC249ADCD08",
                    worker_init=f"""
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw
module list
cd $PBS_O_WORKDIR
hostname
pwd

# Make the hostfiles for each worker
mkdir -p /tmp/hostfiles/
split --lines={nodes_per_cp2k} -d --suffix-length=2 $PBS_NODEFILE /tmp/hostfiles/local_hostfile.
ls /tmp/hostfiles

# Load anaconda
conda activate /lus/grand/projects/CSC249ADCD08/ExaMol/env-polaris
which python""",
                    walltime="12:00:00",
                    queue="preemptable",
                    scheduler_options="#PBS -l filesystems=home:eagle:grand",
                    launcher=SimpleLauncher(),
                    select_options="ngpus=4",
                    nodes_per_block=nodes_per_block,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    cpus_per_node=64,
                    cmd_timeout=120,
                    parallelism=nodes_per_block  # Better reflect workers/block
                )),
            ]
    )
    sim = ASESimulator(
        scratch_dir='cp2k-files',
        optimization_steps=100,
        cp2k_command=f'mpiexec -n {nodes_per_cp2k * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                     f'--hostfile /tmp/hostfiles/local_hostfile.`printf %02d $PARSL_WORKER_RANK` '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/set_affinity_gpu_polaris.sh '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp',
    )
    return config, sim, max_blocks * cp2k_per_block * (prefetch + 1), ['cp2k_b3lyp_svp']


def make_polaris_debug_config() -> tuple[Config, ASESimulator, int, list[str]]:
    """Make a configuration that will run computations on Polaris"""

    # Settings
    max_blocks = 1
    nodes_per_cp2k = 1
    cp2k_per_block = 2
    prefetch = 0

    # Make the config
    config = Config(
        retries=8,
        executors=[
            HighThroughputExecutor(
                max_workers=cp2k_per_block,
                address=address_by_interface('bond0'),
                prefetch_capacity=prefetch,
                provider=PBSProProvider(
                account="CSC249ADCD08",
                    worker_init=f"""
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw
module list
cd $PBS_O_WORKDIR
hostname
pwd

# Make the hostfiles for each worker
mkdir -p /tmp/hostfiles/
split --lines={nodes_per_cp2k} -d --suffix-length=2 $PBS_NODEFILE /tmp/hostfiles/local_hostfile.
ls /tmp/hostfiles

# Load anaconda
conda activate /lus/grand/projects/CSC249ADCD08/ExaMol/env-polaris
which python""",
                    walltime="1:00:00",
                    queue="debug",
                    scheduler_options="#PBS -l filesystems=home:eagle:grand",
                    launcher=SimpleLauncher(),
                    select_options="ngpus=4",
                    nodes_per_block=cp2k_per_block * nodes_per_cp2k,
                    min_blocks=0,
                    max_blocks=max_blocks,
                    cpus_per_node=64,
                    cmd_timeout=120,
                ),
            )]
    )
    sim = ASESimulator(
        scratch_dir='cp2k-files',
        cp2k_command=f'mpiexec -n {nodes_per_cp2k * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                     f'--hostfile /tmp/hostfiles/local_hostfile.`printf %02d $PARSL_WORKER_RANK` '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/set_affinity_gpu_polaris.sh '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp',
    )
    return config, sim, max_blocks * cp2k_per_block * (prefetch + 1), ['cp2k_b3lyp_svp']
