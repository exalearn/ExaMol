"""Algorithms that work particularly quickly"""
from random import sample

from examol.start.base import Starter


class RandomStarter(Starter):
    """Select entries randomly"""

    def _select(self, to_select: list[str], count: int) -> list[str]:
        return sample(to_select, count)
