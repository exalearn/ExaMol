"""Evaluate the molecules from a list prepared by Naveen Dandu"""
from examol.simulate.ase import ASESimulator
from examol.simulate.base import SimResult
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_interface
from parsl.launchers import SimpleLauncher, SrunLauncher
from parsl.providers import PBSProProvider, SlurmProvider
from parsl.app.python import PythonApp
from parsl import Config
from concurrent.futures import as_completed
from argparse import ArgumentParser
from base64 import b64encode, b64decode
from pathlib import Path
from tqdm import tqdm
import pickle as pkl
import pandas as pd
import parsl
import json

if __name__ == "__main__":
    # Parse the inputs
    parser = ArgumentParser()
    parser.add_argument('--system', choices=['bebop', 'local', 'polaris'], help='The system on which we are running')
    parser.add_argument('--configuration', default='cp2k_blyp_szv', help='Name of the ExaMol configuration')
    parser.add_argument('--nodes-per-task', default=1, type=int, help='Number of nodes to use per task')
    parser.add_argument('--num-parallel', default=2, type=int, help='Number of nodes to run in parallel')
    parser.add_argument('--max-to-run', default=None, type=int, help='Maximum number of tasks to run')
    parser.add_argument('--solvent', default='acn', help='Which solvent to use')
    args = parser.parse_args()
    config_name = args.configuration

    # Make the ASE calculator
    #  Call with a local host file defined by the worker's rank
    #  See: https://docs.alcf.anl.gov/polaris/queueing-and-running-jobs/example-job-scripts/#multi-node-ensemble-calculations-example
    cwd = Path().cwd().absolute()
    sim = ASESimulator(
        scratch_dir='ase-files',
        ase_db_path='data.db',
        gaussian_command='g16',  # Hard-coded for Bebop
        cp2k_command=f'mpiexec -n {args.nodes_per_task * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                     f'--hostfile /tmp/hostfiles/$PBS_JOBID/local_hostfile.`printf %02d $((PARSL_WORKER_RANK+1))` '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/set_affinity_gpu_polaris.sh '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp',  # Hard-coded for polaris for now

    )

    # Load in the geometries
    optimizations = pd.read_json('optimization.json', lines=True)
    optimizations.query(f'config_name=="{config_name}"', inplace=True)
    print(f'Loaded {len(optimizations)} optimized geometries for {config_name}')

    # Get the solvation energies which have already ran
    solvation_out = Path('solvation.json')
    if solvation_out.is_file():
        already_ran = set(map(tuple, pd.read_json(solvation_out, lines=True)[["smiles", "charge", "config_name", "solvent"]].values))
    else:
        already_ran = set()

    # Make the parsl configuration
    if args.system == 'polaris':
        htex = HighThroughputExecutor(
            address=address_by_interface('bond0'),
            prefetch_capacity=0,  # Increase if you have many more tasks than workers
            start_method="spawn",  # Needed to avoid interactions between MPI and os.fork
            max_workers=args.num_parallel,
            provider=PBSProProvider(
                account="CSC249ADCD08",
                worker_init=f"""
module reset
module swap PrgEnv-nvhpc PrgEnv-gnu
module load conda
module load cudatoolkit-standalone/11.4.4
module load cray-libsci cray-fftw
module list
cd {cwd}
hostname
pwd
mkdir -p /tmp/hostfiles/$PBS_JOBID
split --lines={args.nodes_per_task} --numeric-suffixes=1 --suffix-length=2 $PBS_NODEFILE /tmp/hostfiles/$PBS_JOBID/local_hostfile.
conda activate /lus/grand/projects/CSC249ADCD08/ExaMol/env""",
                walltime="2:00:00",
                queue="preemptable",
                scheduler_options="#PBS -l filesystems=home:eagle:grand",
                launcher=SimpleLauncher(),
                select_options="ngpus=4",
                nodes_per_block=args.num_parallel * args.nodes_per_task,
                min_blocks=0,
                max_blocks=2,
                cpus_per_node=64,
            ),
        )
    elif args.system == 'bebop':
        htex = HighThroughputExecutor(
            label='bebop_gaussian',
            address=address_by_hostname(),
            max_workers=1,  # Only one task per job
            provider=SlurmProvider(
                partition='knlall',
                launcher=SrunLauncher(),
                nodes_per_block=args.nodes_per_task,
                init_blocks=args.num_parallel,
                min_blocks=0,
                max_blocks=args.num_parallel,
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

    else:
        raise ValueError(f'System not recognized')

    config = Config(
        retries=1,
        executors=[htex],
    )
    parsl.load(config)

    # Make the function to run
    app = PythonApp(sim.compute_energy)

    # Loop over all of them
    futures = []
    for _, row in optimizations.iterrows():
        if (row["smiles"], row["charge"], config_name, args.solvent) in already_ran:
            continue

        # Upack the result to get the xyz
        result: SimResult = pkl.loads(b64decode(row['result']))[0]  # second part is metadata

        future = app(row['smiles'], result.xyz, charge=row['charge'], solvent=args.solvent, config_name=config_name)
        future.info = {'filename': row['filename'], 'smiles': row['smiles'], 'charge': row['charge']}
        futures.append(future)

        # Stop submitting if hit max
        if args.max_to_run is not None and len(futures) == args.max_to_run:
            break

    # Write them out as they complete
    for future in tqdm(as_completed(futures), total=len(futures)):
        exc = future.exception()
        if exc is not None:
            with open('solv-failures.json', 'a') as fp:
                print(json.dumps({**future.info, 'exception': str(exc)}), file=fp)
            continue

        with open(solvation_out, 'a') as fp:
            res = future.result()
            print(json.dumps({
                **future.info,
                'system': args.system,
                'config_name': config_name,
                'solvent': args.solvent,
                'energy': res[0].energy,
                'result': b64encode(pkl.dumps(res)).decode('ascii'),
            }), file=fp)
