"""Base classes for storage utilities"""
import gzip
from abc import ABC
from pathlib import Path
from typing import Iterable
from contextlib import AbstractContextManager

from examol.store.models import MoleculeRecord
from examol.utils.chemistry import get_inchi_key_from_molecule_string


class MoleculeStore(AbstractContextManager, ABC):
    """Base class defining how to interface with a dataset of molecule records.

    Data stores provide the ability to persist the data collected by ExaMol to disk during a run.
    The :meth:`update_record` call need not immediately persist the data but should ensure that the data
    is stored on disk eventually.
    In fact, it is actually better for the update operation to not block until the resulting write has completed.

    Stores do not need support concurrent access from multiple client, which is why this documentation avoids the word "database."
    """

    def __getitem__(self, mol_key: str) -> MoleculeRecord:
        """Retrieve a molecule record"""
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def __contains__(self, item: str | MoleculeRecord):
        raise NotImplementedError()

    def get_or_make_record(self, mol_string: str) -> MoleculeRecord:
        """Either the existing record for a molecule or make a new one

        Args:
            mol_string: String describing a molecule (e.g., SMILES string)
        Returns:
            Record
        """
        key = get_inchi_key_from_molecule_string(mol_string)
        if key not in self:
            record = MoleculeRecord.from_identifier(mol_string)
            self.update_record(record)
            return record
        else:
            return self[key]

    def iterate_over_records(self) -> Iterable[MoleculeRecord]:
        """Iterate over all records in data

        Yields:
            A single record
        """
        raise NotImplementedError()

    def update_record(self, record: MoleculeRecord):
        """Update a single record

        Args:
            record: Record to be updated
        """
        raise NotImplementedError()

    def update_records(self, records: Iterable[MoleculeRecord]):
        """Update many records at once

        Args:
            records: Iterator over records to be stored
        """
        for record in records:
            self.update_record(record)

    def export_records(self, path: Path):
        """Save a current copy of the database to disk as line-delimited JSON

        Args:
            path: Path in which to save all data. Use a ".json.gz"
        """

        with (gzip.open(path, 'wt') if path.name.endswith('.gz') else open(path, 'w')) as fp:
            for record in self.iterate_over_records():
                print(record.json(), file=fp)
