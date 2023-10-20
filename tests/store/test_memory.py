"""Test an in-memory store for molecular data"""

from pytest import fixture

from examol.store.db.memory import InMemoryStore
from examol.store.models import MoleculeRecord


@fixture()
def records() -> list[MoleculeRecord]:
    return [MoleculeRecord.from_identifier(smiles) for smiles in ['C', 'O', 'N']]


def test_store(tmpdir, records):
    # Open the database
    db_path = tmpdir / 'db.json.gz'
    with InMemoryStore(db_path) as store:
        assert len(store) == 0

        # Add the records
        for record in records:
            store.update_record(record)
        assert len(store) == 3
        assert MoleculeRecord.from_identifier('C') in store
        assert MoleculeRecord.from_identifier('Br') not in store

    # Load database back in
    with InMemoryStore(db_path) as store:
        assert len(store) == 3
