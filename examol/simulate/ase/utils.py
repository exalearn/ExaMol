"""Utilities related to using ASE"""
from contextlib import contextmanager
from typing import Iterator

import ase
import numpy as np
from ase.calculators.calculator import Calculator


@contextmanager
def make_ephemeral_calculator(calc: Calculator | dict) -> Iterator[Calculator]:
    """Make a calculator then tear it down after completion

    Args:
        calc: Already-defined calculator or a dict defining it.
            The dict must contain the key "name" to define the name
            of the code and could contain the keys "args" and "kwargs"
            to define the arguments and keyword arguments for creating
            a new one, respectively.

    Yields:
        An Calculator that is town down as the context manager exits
    """

    # Special case: just yield the calculator
    if isinstance(calc, Calculator):
        yield calc
        return

    # Get the arguments for the calculator
    args = calc.get('args', [])
    kwargs = calc.get('kwargs', {})

    # Otherwise, create one
    name = calc['name'].lower()
    if name == 'cp2k':
        from ase.calculators.cp2k import CP2K
        calc = CP2K(*calc.get('args', []), **calc.get('kwargs', {}))
        yield calc

        # Kill the calculator by deleting the object to stop the underlying
        #  shell and then set the `_shell` parameter of the object so that the
        #  calculator object's destructor will skip the shell shutdown process
        #  when the object is finally garbage collected
        calc.__del__()
        calc._shell = None
    elif name == 'xtb':
        from xtb.ase.calculator import XTB
        yield XTB(*args, **kwargs)
    else:
        raise ValueError('No such calculator')


def buffer_cell(atoms, buffer_size: float = 6.):
    """Buffer the cell such that it has a vacuum layer around the side

    Args:
        atoms: Atoms to be centered
        buffer_size: Size of the buffer to place around the atoms
    """

    atoms.positions -= atoms.positions.min(axis=0)
    atoms.cell = atoms.positions.max(axis=0) + buffer_size * 2
    atoms.positions += atoms.cell.max(axis=0) / 2 - atoms.positions.mean(axis=0)


def initialize_charges(atoms: ase.Atoms, charge: int):
    """Set initial charges to sum up to a certain value

    Args:
         atoms: Atoms object to be manipulated
         charge: Total charge for the system
    """
    charges = np.ones((len(atoms),)) * (charge / len(atoms))
    atoms.set_initial_charges(charges)
