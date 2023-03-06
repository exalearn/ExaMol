from pytest import fixture
from ase.build import molecule

from examol.simulate.base import SimResult
from examol.store.models import MoleculeRecord
from examol.utils.conversions import write_to_string


@fixture()
def sim_result() -> SimResult:
    mol = molecule('CH4')
    return SimResult(
        xyz=write_to_string(mol, 'xyz'),
        charge=0,
        energy=-1,
        config_name='test',
        solvent=None
    )


@fixture()
def record() -> MoleculeRecord:
    return MoleculeRecord.from_identifier('C')
