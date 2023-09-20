"""Database-related reporting"""
import logging

from .base import BaseReporter
from ..steer.base import MoleculeThinker

logger = logging.getLogger(__name__)


class DatabaseWriter(BaseReporter):
    """Writes the current database to disk as a JSON file"""

    def report(self, thinker: MoleculeThinker):
        temp_database = thinker.database.copy()
        with open(thinker.run_dir / 'database.json', 'w') as fp:
            for record in temp_database.values():
                print(record.to_json(), file=fp)
        logger.info(f'Saved {len(thinker.database)} records to disk')
