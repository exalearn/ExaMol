"""Algorithms that work particularly quickly"""
from random import sample
from typing import Iterable

from examol.start.base import Starter
from more_itertools import take


class RandomStarter(Starter):
    """Select entries randomly"""

    def select(self, to_select: Iterable[str]) -> list[str]:
        if self.max_to_consider is not None:
            pool = take(self.max_to_consider, to_select)
        else:
            pool = list(to_select)
        return sample(pool, self.to_select)
