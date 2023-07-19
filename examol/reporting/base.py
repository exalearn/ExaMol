"""Base class for reporting implementations"""
from dataclasses import dataclass
from threading import Thread

from examol.steer.base import MoleculeThinker


@dataclass()
class BaseReporter:
    """Base class for all reporter functions"""

    def monitor(self, thinker: MoleculeThinker, frequency: float = 10) -> Thread:
        """Continually report on the progress of the thinker in a separate thread

        The thread runs until the thinker halts and is launched as a daemon thread
        so that it will not prevent the simulation from exiting properly.

        Args:
            thinker: Thinker to be monitored
            frequency: How many seconds between report
        Returns:
            Thread that holds the monitor
        """

        # Make the function to be run
        def _to_run():
            while not thinker.done.wait(frequency):
                self.report(thinker)
            self.report(thinker)  # Final report before exiting

        thread = Thread(target=_to_run, daemon=True, name=f'monitor_{self.__class__.__name__.lower()}')
        thread.start()
        return thread

    def report(self, thinker: MoleculeThinker):
        """Generate a report for the status of the thinker

        Args:
            thinker: Thinker to report about
        """
        raise NotImplementedError()
