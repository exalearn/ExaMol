"""Evaluate the molecules from a list prepared by Naveen Dandu"""
from examol.simulate.ase import ASESimulator
from examol.simulate.base import SimResult
from examol.simulate.initialize import generate_inchi_and_xyz
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname
from parsl.launchers import SimpleLauncher
from parsl.providers import PBSProProvider
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
import os

if __name__ == "__main__":
    # Parse the inputs
    parser = ArgumentParser()
    parser.add_argument('--cp2k-configuration', default='cp2k_blyp_szv')
    parser.add_argument('--nodes-per-cp2k', default=1, type=int)
    parser.add_argument('--num-parallel', default=2, type=int)
    parser.add_argument('--max-to-run', default=None, type=int)
    parser.add_argument('--solvent', default='acn', help='Which solvent to use')
    args = parser.parse_args()
    config_name = args.cp2k_configuration

    # Make the ASE calculator
    #  Call with a local host file defined by the worker's rank
    #  See: https://docs.alcf.anl.gov/polaris/queueing-and-running-jobs/example-job-scripts/#multi-node-ensemble-calculations-example
    cwd = Path().cwd().absolute()
    sim = ASESimulator(
        scratch_dir='cp2k-files',
        cp2k_buffer=6.0,
        cp2k_command=f'mpiexec -n {args.nodes_per_cp2k * 4} --ppn 4 --cpu-bind depth --depth 8 -env OMP_NUM_THREADS=8 '
                     f'--hostfile {cwd}/hostfiles/$PBS_JOBID/local_hostfile.{args.cp2k_configuration}.`printf %02d $((PARSL_WORKER_RANK+1))` '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/set_affinity_gpu_polaris.sh '
                     '/lus/grand/projects/CSC249ADCD08/cp2k/cp2k-git/exe/local_cuda/cp2k_shell.psmp',
    )

    # Load in the geometries
    optimizations = pd.read_json('output.json', lines=True)
    optimizations.query(f'config_name=="{config_name}"', inplace=True)
    print(f'Loaded {len(optimizations)} optimized geometries for {config_name}')

    # Get the solvation energies which have already ran
    solvation_out = Path('solvation.json')
    if solvation_out.is_file():
        already_ran = set(map(tuple, pd.read_json(solvation_out, lines=True)[["smiles", "charge", "config_name", "solvent"]].values))
    else:
        already_ran = set()

    # Make the parsl configuration
    config = Config(
        retries=1,
        executors=[
            HighThroughputExecutor(
                address=address_by_hostname(),
                prefetch_capacity=0,  # Increase if you have many more tasks than workers
                start_method="fork",  # Needed to avoid interactions between MPI and os.fork
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
mkdir -p hostfiles/$PBS_JOBID
split --lines={args.nodes_per_cp2k} --numeric-suffixes=1 --suffix-length=2 $PBS_NODEFILE hostfiles/$PBS_JOBID/local_hostfile.{args.cp2k_configuration}.
conda activate /lus/grand/projects/CSC249ADCD08/ExaMol/env""",
                    walltime="6:00:00",
                    queue="preemptable",
                    scheduler_options="#PBS -l filesystems=home:eagle:grand",
                    launcher=SimpleLauncher(),
                    select_options="ngpus=4",
                    nodes_per_block=args.num_parallel * args.nodes_per_cp2k,
                    min_blocks=0,
                    max_blocks=10,
                    cpus_per_node=64,
                ),
            ),
        ]
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

        future = app(result.xyz, charge=row['charge'], solvent=args.solvent, config_name=config_name)
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
                'config_name': config_name,
                'solvent': args.solvent,
                'energy': res[0].energy,
                'result': b64encode(pkl.dumps(res)).decode('ascii'),
            }), file=fp)
