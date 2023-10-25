"""Relax the surfaces in parallel on HPC"""
from argparse import ArgumentParser
from pathlib import Path
import logging
import sys

from ase.io import read

from examol.simulate.ase import ASESimulator
from examol.utils.conversions import write_to_string

if __name__ == "__main__":
    # Make the argument parser
    parser = ArgumentParser()
    parser.add_argument('--config', default='cp2k_pbe_dzvp', help='Configuration to use for computations')
    parser.add_argument('initial', nargs='+', help='Directory holding initial structures to relax', type=Path)
    args = parser.parse_args()

    # Make the logger
    logger = logging.getLogger('main')
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Find the structures to relax
    all_surfaces = sum([list(Path(i).rglob('unrelaxed.extxyz')) for i in args.initial], [])
    logger.info(f'Found {len(all_surfaces)} surfaces')
    unrelaxed_surfaces = set(
        surface for surface in all_surfaces
        if not surface.parent.joinpath('relaxed.extxyz').is_file()
    )
    logger.info(f'Found {len(unrelaxed_surfaces)} we have not relaxed yet')

    # Create the simulator
    sim = ASESimulator(
        scratch_dir='./scratch',
        cp2k_command='/home/lward/Software/cp2k-2023.2/exe/local_cuda/cp2k_shell.ssmp',
        clean_after_run=False
    )

    # Loop over the surfaces to relax
    #  TODO (wardlt): Use Parsl to run loop on HPC
    for surface in unrelaxed_surfaces:
        logger.info(f'Starting to relax {surface.parent}')
        # Load the file and write as an extended XYZ file
        atoms = read(surface)
        xyz = write_to_string(atoms, 'extxyz', columns=['symbols', 'positions', 'move_mask'])

        # Relax the surfaces
        run_name = f'{surface.parent.parent.name}-{surface.parent.name}'
        result, _, _ = sim.optimize_structure(run_name, xyz, config_name=args.config, charge=0, solvent=None)

        # Save the relaxed structure
        out_path = surface.parent.joinpath('relaxed.extxyz')
        result.atoms.write(out_path, columns=['symbols', 'positions', 'move_mask'])
        logger.info(f'Done with {surface.parent}. Saved result to {out_path}')
