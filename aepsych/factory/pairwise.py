# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

from typing import List, Tuple, Union

import gpytorch

from aepsych.config import Config
from aepsych.factory.default import (
    _get_default_cov_function,
    _get_default_mean_function,
    default_mean_covar_factory,
)
from aepsych.kernels.pairwisekernel import PairwiseKernel


def pairwise_mean_covar_factory(
    config: Config,
) -> Tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:
    lb = config.gettensor("common", "lb")
    ub = config.gettensor("common", "ub")
    assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
    assert lb.shape[0] >= 2, "PairwiseKernel requires at least 2 dimensions!"
    config_dim = lb.shape[0]

    shared_dims: Union[List[int], None] = config.getlist(
        "pairwise_mean_covar_factory", "shared_dims", fallback=None
    )
    if shared_dims is not None:
        shared_dims = [int(d) for d in shared_dims]
        assert len(shared_dims) < config_dim, "length of shared_dims must be < dim!"
        for dim in shared_dims:
            assert dim < len(shared_dims)
    else:
        shared_dims = []

    base_mean_covar_factory = config.getobj(
        "pairwise_mean_covar_factory",
        "base_mean_covar_factory",
        fallback=default_mean_covar_factory,
    )

    if base_mean_covar_factory is not default_mean_covar_factory:
        raise NotImplementedError(
            "Only default_mean_covar_factory is supported for the base factor of pairwise_mean_covar_factory right now!"
        )

    if len(shared_dims) > 0:
        active_dims = [i for i in range(config_dim) if i not in shared_dims]
        assert (
            len(active_dims) % 2 == 0
        ), "dimensionality of non-shared dims must be even!"
        mean = _get_default_mean_function(config)
        cov1 = _get_default_cov_function(
            config, len(active_dims) // 2, stimuli_per_trial=1
        )

        cov2 = _get_default_cov_function(
            config, len(shared_dims), active_dims=shared_dims, stimuli_per_trial=1
        )

        covar = PairwiseKernel(cov1, active_dims=active_dims) * cov2

    else:
        assert config_dim % 2 == 0, "dimensionality must be even!"
        mean = _get_default_mean_function(config)
        cov = _get_default_cov_function(config, config_dim // 2, stimuli_per_trial=1)
        covar = PairwiseKernel(cov)

    return mean, covar
