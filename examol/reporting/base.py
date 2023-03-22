"""Base class for reporting implementations"""
from examol.steer.base import MoleculeThinker


class BaseReporter:
    """Base class for all reporter functions"""

    def report(self, thinker: MoleculeThinker):
        """Generate a report for the status of the thinker

        Args:
            thinker: Thinker to report about
        """
        raise NotImplementedError()
