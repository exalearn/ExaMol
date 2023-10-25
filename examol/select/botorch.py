"""Employ the acquisition functions from `BOTorch <https://botorch.org/>`_"""
from typing import List, Any, Callable, Sequence, Optional

from botorch.acquisition.multi_objective import qExpectedHypervolumeImprovement
from botorch.acquisition.objective import PosteriorTransform
from botorch.posteriors import Posterior, GPyTorchPosterior
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions import FastNondominatedPartitioning
from gpytorch.distributions import MultitaskMultivariateNormal

try:
    from botorch.acquisition import AcquisitionFunction
except ImportError as e:  # pragma: no-coverage
    raise ImportError('You may need to install BOTorch and PyTorch') from e
from botorch.models.model import Model
from torch import Tensor
import numpy as np
import torch

from examol.select.base import RankingSelector, _extract_observations
from examol.store.models import MoleculeRecord
from examol.store.recipes.base import PropertyRecipe


class _EnsembleCovarianceModel(Model):
    """Model which generates a multivariate Gaussian distribution given samples from an ensemble of models"""

    def __init__(self, num_outputs: int):
        super().__init__()
        self._num_outputs = num_outputs

    @property
    def num_outputs(self) -> int:
        return self._num_outputs

    def posterior(
            self,
            X: Tensor,
            output_indices: Optional[List[int]] = None,
            observation_noise: bool = False,
            posterior_transform: Optional[PosteriorTransform] = None,
            **kwargs: Any,
    ) -> Posterior:
        # Reshape X from (b x q x d) -> (b x q x o x s):
        #  b - batch size used for efficiency's sake
        #  q - number of points considered together (this is the batch size for active learning's sake)
        #  o - number of outputs
        #  s - ensemble size
        b, q, _ = X.size()
        x_reshaped = torch.unflatten(X, dim=-1, sizes=(self._num_outputs, -1))

        # Compute the mean of each dimension
        #   GPyTorch wants this as a (b x q x o) matrix
        means = torch.mean(x_reshaped, axis=-1)

        # Compute the covariance between points in a batch (q)
        #  GPyTorch wants this as a (b x (q x o)) matrix
        #  Thanks: https://github.com/pytorch/pytorch/issues/19037
        centered = x_reshaped - means.unsqueeze(-1)  # b x q x o x s
        combined_task_and_samples = torch.flatten(centered, start_dim=1, end_dim=2)  # b x (q x o) x s
        d = combined_task_and_samples.shape[-1]
        cov = 1 / (d - 1) * combined_task_and_samples @ combined_task_and_samples.transpose(-1, -2)

        # Make the multivariate normal as an output
        posterior = GPyTorchPosterior(
            distribution=MultitaskMultivariateNormal(mean=means, covariance_matrix=cov, validate_args=True)
        )
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior


class BOTorchSequentialSelector(RankingSelector):
    """Use an acquisition function from BOTorch to score candidates assuming a pool size of :math:`q=1`

    Provide the acquisition function type and any options needed to configure it.
    Options can be updated by supplying a function which updates them based
    on the properties of molecules which have been evaluated so far.

    For example, Expected Improvement which updates the maximum observed value would be

    .. code-block:: python

        def update_fn(selector: 'BOTorchSequentialSelector', obs: np.ndarray) -> dict:
            return {'best_f': max(obs) if selector.maximize else min(obs)}

        selector = BOTorchSequentialSelector(qExpectedImprovement,
                                             acq_options={'best_f': 0.5},
                                             acq_options_updater=update_fn,
                                             to_select=1)

    Args:
        acq_function_type: Class of the acquisition function
        acq_options: Dictionary of options passed to the acquisition function maker
        acq_options_updater: Function which takes the current selector and an array of observations
            of shape (num molecules) x (num recipes)
        maximize: Whether to maximize or minimize the objectives
        to_select: Number of top candidates to select each round
    """

    multiobjective = True

    def __init__(self,
                 acq_function_type: type[AcquisitionFunction],
                 acq_options: dict[str, object],
                 to_select: int,
                 acq_options_updater: Callable[['BOTorchSequentialSelector', np.ndarray], dict] | None = None,
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
            self.acq_options = self.acq_options_updater(self, outputs)

        self.acq_function = self.acq_function_type(model=_EnsembleCovarianceModel(len(recipes)), **self.acq_options)

    def _assign_score(self, samples: np.ndarray) -> np.ndarray:
        # Shape the tensor in the form expected by BOtorch's GPyTorch
        #  Samples is a `objectives x samples x models` array
        #  BOTorch expects `samples x batch size (q) x (objectives x models)` array
        samples_tensor = samples[:, :, None, :]  # o x s x q x m
        samples_tensor = samples_tensor.transpose((1, 2, 0, 3))  # `s x q x o x m`
        samples_tensor = torch.from_numpy(samples_tensor)
        samples_tensor = torch.flatten(samples_tensor, start_dim=2)  # s x q x (o x m)
        score_tensor = self.acq_function(samples_tensor)
        return score_tensor.detach().cpu().numpy()


def _evhi_update_fn(selector: 'EHVISelector', obs: np.ndarray) -> dict:
    """Options update function used for EVHI

    Args:
        selector: Selector
        obs: Observations of target properties
    """
    # Determine the reference point based on the observations
    #  We should use the minim
    options = selector.acq_options.copy()
    if not isinstance(selector.maximize, bool):
        obs = obs.copy()
        for i, m in enumerate(selector.maximize):
            if not m:
                obs[:, i] *= -1
    elif not selector.maximize:
        obs = obs * -1
    options['ref_point'] = obs.min(axis=0)

    # Set up the partitioning
    options['partitioning'] = FastNondominatedPartitioning(
        ref_point=torch.from_numpy(options['ref_point']),
        Y=torch.from_numpy(obs),
    )
    return options


class EHVISelector(BOTorchSequentialSelector):
    """Rank entries based on the Expected Hypervolume Improvement (EVHI)

    EVHI is a multi-objective optimization scores which measures how much a new point will expand the Pareto surface.
    We use the Monte Carlo implementation of EVHI of `Daulton et al. <https://arxiv.org/abs/2006.05078>`_,
    but do not yet support the algorithms batch-aware implementation.

    Constructing the Pareto surface requires the definition of a reference point where
    farther from the reference point is better.
    We use the minimum value of objectives which are being maximized and the maximum value of those being minimized.

    Args:
        maximize: Whether to maximize or minimize the objectives
        to_select: Number of top candidates to select each round
    """

    def __init__(self, to_select: int, maximize: bool | Sequence[bool] = True):
        super().__init__(
            acq_function_type=qExpectedHypervolumeImprovement,
            acq_options_updater=_evhi_update_fn,
            acq_options={'sampler': SobolQMCNormalSampler(sample_shape=torch.Size([128]))},  # TODO (wardlt): Make sampler configurable
            to_select=to_select,
            maximize=maximize
        )
