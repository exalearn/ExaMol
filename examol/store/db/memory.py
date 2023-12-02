"""Stores that keep the entire dataset in memory"""
import gzip
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path
from time import monotonic, sleep
from threading import Event
from typing import Iterable

from examol.store.db.base import MoleculeStore
from examol.store.models import MoleculeRecord

logger = logging.getLogger(__name__)


class InMemoryStore(MoleculeStore):
    """Store all molecule records in memory, write to disk as a single file

    The class will start checkpointing as soon as any record is updated
    but no more frequently than :attr:`write_freq`

    Args:
        path: Path from which to read data. Must be a JSON file, can be compressed with GZIP.
            Set to ``None`` if you do not want data to be stored
        write_freq: Minimum time between writing checkpoints
    """

    def __init__(self, path: Path | None, write_freq: float = 10.):
        self.path = Path(path) if path is not None else path
        self.write_freq = write_freq
        self.db: dict[str, MoleculeRecord] = {}

        # Start thread which writes until
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        self._write_thread: Future | None = None
        self._updates_available: Event = Event()
        self._closing = Event()

        # Start by loading the molecules
        self._load_molecules()

    def __enter__(self):
        if self.path is not None:
            logger.info('Start the writing thread')
            self._write_thread = self._thread_pool.submit(self._writer)

            # Add a callback to print a logging message if there is an error
            def _write_if_error(future: Future):
                if (exc := future.exception()) is not None:
                    logger.warning(f'Write thread failed: {exc}')
                logger.info('Write thread has exited')

            self._write_thread.add_done_callback(_write_if_error)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path is not None:
            # Trigger a last write
            logger.info('Triggering a last write to the database')
            self._closing.set()
            self._write_thread.result()

            # Mark that we're closed
            self._write_thread = None
            self._closing.clear()

    def _load_molecules(self):
        """Load molecules from disk"""
        if self.path is None or not self.path.is_file():
            return
        logger.info(f'Loading data from {self.path}')
        with (gzip.open(self.path, 'rt') if self.path.name.endswith('.gz') else self.path.open()) as fp:
            for line in fp:
                record = MoleculeRecord.parse_raw(line)
                self.db[record.key] = record
        logger.info(f'Loaded {len(self.db)} molecule records')

    def iterate_over_records(self) -> Iterable[MoleculeRecord]:
        yield from list(self.db.values())

    def __getitem__(self, item):
        return self.db[item]

    def __len__(self):
        return len(self.db)

    def __contains__(self, item: str | MoleculeRecord):
        mol_key = item.key if isinstance(item, MoleculeRecord) else item
        return mol_key in self.db

    def _writer(self):
        next_write = 0
        while self._updates_available.is_set() or not self._closing.is_set():
            # Wait until updates are available and the standoff is not met, or if we're closing
            while monotonic() < next_write or not self._closing.is_set():
                if self._updates_available.wait(timeout=1):  # Check for termination condition once per second
                    to_sleep = next_write - monotonic()
                    if to_sleep > 0:
                        sleep(to_sleep)
                    break

            # Mark that we've caught up with whatever signaled this thread
            self._updates_available.clear()

            # Checkpoint and advance the standoff
            temp_path = self.path.parent / (self.path.name + "-new")
            self.export_records(temp_path)
            move(temp_path, self.path)
            next_write = monotonic() + self.write_freq

    def update_record(self, record: MoleculeRecord):
        self.db[record.key] = record
        self._updates_available.set()
