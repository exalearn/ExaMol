"""Configurations for the different computational environments

Add a new environment by creating a function which produces a Simulation tool,
Parsl config for the desired environment,
and a list of the energy configurations it supports
"""
from parsl.executors import HighThroughputExecutor
from parsl.providers import CobaltProvider
from parsl.launchers import AprunLauncher
from parsl.addresses import address_by_interface
from parsl.config import Config

from examol.simulate.ase import ASESimulator


def make_local_config() -> tuple[Config, ASESimulator, list[str]]:
    """Make a configuration that will run computations locally"""

    config = Config(
        executors=[HighThroughputExecutor(max_workers=1, address='127.0.0.1')]
    )
    sim = ASESimulator(scratch_dir='run_data')
    return config, sim, ['xtb', 'mopac_pm7']


def make_theta_config() -> tuple[Config, ASESimulator, list[str]]:
    """Make a configuration that will run computations locally"""

    config = Config(
        executors=[
            HighThroughputExecutor(
                max_workers=4,
                cpu_affinity='block',
                address=address_by_interface('vlan2360'),
                provider=CobaltProvider(
                    queue='debug-flat-quad',
                    account='CSC249ADCD08',
                    launcher=AprunLauncher(overrides="-d 64 --cc depth"),
                    walltime='00:30:00',
                    nodes_per_block=2,
                    max_blocks=1,
                    worker_init='''module load conda; source activate /lus/grand/projects/CSC249ADCD08/ExaMol/env-theta''',
                    cmd_timeout=120,
                ),
            )]
    )
    sim = ASESimulator(scratch_dir='run_data')
    return config, sim, ['xtb', 'mopac_pm7']
