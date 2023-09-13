"""Configurations for the different computational environments

Add a new environment by creating a function which produces a Simulation tool,
Parsl config for the desired environment,
the number of execution slots,
and a list of the energy configurations it supports
"""
from parsl.executors import HighThroughputExecutor
from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
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

    max_blocks = 8
    prefetch = 16
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
    return config, sim, max_blocks * 128 * prefetch, ['xtb', 'mopac_pm7']


def make_theta_debug_config() -> tuple[Config, ASESimulator, int, list[str]]:
    """Make a configuration that will run computations on a small job"""

    max_blocks = 1
    prefetch = 4
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
    return config, sim, max_blocks * 4 * prefetch, ['xtb']