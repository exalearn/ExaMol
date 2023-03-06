"""Utilities related to using ASE"""
from contextlib import contextmanager
from typing import Iterator

from ase.calculators.calculator import Calculator


@contextmanager
def make_ephemeral_calculator(calc: Calculator | dict) -> Iterator[Calculator]:
    """Make a calculator then tear it down after completion

    Args:
        calc: Already-defined calculatori or a dict defining it.
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

    # Otherwise, create one
    name = calc['name']
    if name.lower() == 'cp2k':
        from ase.calculators.cp2k import CP2K
        calc = CP2K(*calc.get('args', []), **calc.get('kwargs', {}))
        yield calc

        # Kill the calculator by deleting the object to stop thei underlying
        #  shell and then set the `_shell` parameter of the object so that the
        #  calculator object's destructor will skip the shell shutdown process
        #  when the object is finally garbage collected
        calc.__del__()
        calc._shell = None
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
