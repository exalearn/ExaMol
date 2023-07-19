"""Employ the acquisition functions from `BOTorch <https://botorch.org/>`_"""
from typing import List, Any, Callable, Sequence

try:
    from botorch.acquisition import AcquisitionFunction
except ImportError as e:  # pragma: no-coverage
    raise ImportError('You may need to install BOTorch and PyTorch') from e
from botorch.models.ensemble import EnsembleModel
from botorch.models.model import Model
from torch import Tensor
import numpy as np
import torch

from examol.select.base import RankingSelector, _extract_observations
from examol.store.models import MoleculeRecord
from examol.store.recipes import PropertyRecipe


class _ExternalEnsembleModel(EnsembleModel):
    """Routes outputs from another modeling program through the BOTorch
    :class:`~botorch.models.ensemble.EnsembleModel` interface

    This model takes the outputs of the other program as input and returns them as a Tensor,
    which ``EnsembleModel`` will transform to the Posterior class needed by BOTorch.
    """

    def __init__(self, num_outputs=1):
        super().__init__()
        self._num_outputs = num_outputs

    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs: Any) -> Model:
        return self

    def subset_output(self, idcs: List[int]) -> Model:
        raise NotImplementedError()

    def forward(self, X: Tensor) -> Tensor:
        return X


class BOTorchSequentialSelector(RankingSelector):
    """Use an acquisition function from BOTorch to score candidates assuming a pool size of :math:`q=1`

    Provide the acquisition function type and any options needed to configure it.
    Options can be updated by supplying a function which updates them based
    on the properties of molecules which have been evaluated so far.

    For example, Expected Improvement which updates the maximum observed value would be

    .. code-block:: python

        def update_fn(obs: np.ndarray, options: dict) -> dict:
            options['best_f'] = max(obs)
            return options

        selector = BOTorchSequentialSelector(qExpectedImprovement,
                                             acq_options={'best_f': 0.5},
                                             acq_options_updater=update_fn,
                                             to_select=1)

    Args:
        acq_function_type: Class of the acquisition function
        acq_options: Dictionary of options passed to the acquisition function maker
        to_select: Number of top candidates to select each round
        acq_options_updater: Function which takes an array of observed outputs, the current dictionary of options,
            and returns an updated set of options.
        maximize: Whether to maximize or minimize the outputs
    """

    def __init__(self,
                 acq_function_type: type[AcquisitionFunction],
                 acq_options: dict[str, object],
                 to_select: int,
                 acq_options_updater: Callable[[np.ndarray, dict], dict] | None = None,
                 maximize: bool = True):
        self.acq_function: AcquisitionFunction | None = None
        self.acq_function_type = acq_function_type
        self.acq_options = acq_options.copy()
        self.acq_options_updater = acq_options_updater
        super().__init__(to_select, maximize)

    def update(self, database: dict[str, MoleculeRecord], recipes: Sequence[PropertyRecipe]):
        if self.acq_options_updater is not None:
            # Run the update function on the properties observed so far
            outputs = _extract_observations(database, recipes)
            self.acq_options = self.acq_options_updater(outputs, self.acq_options)

        self.acq_function = self.acq_function_type(model=_ExternalEnsembleModel(), **self.acq_options)

    def _assign_score(self, samples: np.ndarray) -> np.ndarray:
        # Change sign if need be
        if not self.maximize:
            samples = -1 * samples

        # Shape the tensor in the form expected by BOtorch's EnsemblePosterior
        #  Samples is a `objectives x samples x models` array
        #  BOTorch expects `samples x models x batch size (q) x objectives
        samples_tensor = samples[:, :, None, :]  # Insert a "q" dimension
        samples_tensor = samples_tensor.transpose((1, 3, 2, 0))
        samples_tensor = torch.from_numpy(samples_tensor)  # `(b) x s x q x m`
        score_tensor = self.acq_function(samples_tensor)
        return score_tensor.detach().cpu().numpy()
