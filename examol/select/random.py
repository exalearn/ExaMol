"""Random selection. A useful baseline"""
from random import sample
from typing import Iterator

import numpy as np

from .base import Selector


class RandomSelector(Selector):
    """Select which computations to perform at random"""

    def __init__(self, to_select: int):
        super().__init__(to_select=to_select)
        self._options = set()

    def _add_possibilities(self, keys: list, samples: np.ndarray, **kwargs):
        self._options.update(keys)

    def _dispense(self) -> Iterator[tuple[object, float]]:
        yield from sample(self._options, self.to_select)
