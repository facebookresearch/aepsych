#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
from typing import Dict, List, Optional, Tuple, Type, Union

import gpytorch

import numpy as np
import torch

from aepsych.config import Config
from aepsych.factory.factory import ordinal_mean_covar_factory
from aepsych.likelihoods.ordinal import OrdinalLikelihood
from aepsych.models.base import AEPsychModel
from aepsych.models.ordinal_gp import OrdinalGPModel
from aepsych.models.utils import get_probability_space
from aepsych.utils import get_dim
from botorch.acquisition.objective import PosteriorTransform
from botorch.models import SingleTaskVariationalGP

from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.likelihoods import BernoulliLikelihood, BetaLikelihood
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from botorch.models.utils.inducing_point_allocators import InducingPointAllocator
from gpytorch.kernels import Kernel, MaternKernel, ProductKernel
from gpytorch.likelihoods import BernoulliLikelihood, BetaLikelihood, Likelihood
from gpytorch.means import Mean
from gpytorch.mlls import VariationalELBO
from gpytorch.priors import GammaPrior

from gpytorch.variational import (
    _VariationalDistribution,
    _VariationalStrategy,
    VariationalStrategy,
)
from torch import Tensor


# TODO: Find a better way to do this on the Ax/Botorch side
class MyHackyVariationalELBO(VariationalELBO):
    def __init__(self, likelihood, model, beta=1.0, combine_terms=True):
        num_data = model.model.train_targets.shape[0]
        super().__init__(likelihood, model.model, num_data, beta, combine_terms)


class VariationalGP(AEPsychModel, SingleTaskVariationalGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = False,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
        **kwargs,
    ) -> None:
        if likelihood is None:
            likelihood = self.default_likelihood
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            inducing_point_allocator=inducing_point_allocator,
            **kwargs,
        )

    @classmethod
    def get_mll_class(cls):
        return MyHackyVariationalELBO

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None) -> Dict:
        classname = cls.__name__

        options = super().get_config_options(config, classname)
        inducing_size = config.getint(classname, "inducing_size", fallback=100)
        learn_inducing_points = config.getboolean(
            classname, "learn_inducing_points", fallback=False
        )

        options.update(
            {
                "inducing_points": inducing_size,
                "learn_inducing_points": learn_inducing_points,
            }
        )

        return options

    def posterior(
        self,
        X,
        output_indices=None,
        observation_noise=False,
        posterior_transform: Optional[PosteriorTransform] = None,
        *args,
        **kwargs,
    ) -> GPyTorchPosterior:
        self.eval()  # make sure model is in eval mode

        # input transforms are applied at `posterior` in `eval` mode, and at
        # `model.forward()` at the training time
        X = self.transform_inputs(X)

        # check for the multi-batch case for multi-outputs b/c this will throw
        # warnings
        X_ndim = X.ndim
        if self.num_outputs > 1 and X_ndim > 2:
            X = X.unsqueeze(-3).repeat(*[1] * (X_ndim - 2), self.num_outputs, 1, 1)
        dist = self.model(X)
        if observation_noise:
            dist = self.likelihood(dist, *args, **kwargs)

        posterior = GPyTorchPosterior(distribution=dist)
        if hasattr(self, "outcome_transform"):
            posterior = self.outcome_transform.untransform_posterior(posterior)
        if posterior_transform is not None:
            return posterior_transform(posterior)
        return posterior


class BinaryClassificationGP(VariationalGP):
    stimuli_per_trial = 1
    outcome_type = "binary"
    default_likelihood = BernoulliLikelihood()

    def predict_probability(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance in probability space.

        Args:
            x (torch.Tensor): Points at which to predict from the model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            post = self.posterior(x)

            fmean, fvar = get_probability_space(
                likelihood=self.likelihood, posterior=post
            )

        return fmean, fvar


class MultitaskBinaryClassificationGP(BinaryClassificationGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        task_dims: Optional[List[int]] = None,
        num_tasks: Optional[List[int]] = None,
        ranks: Optional[List[int]] = None,
        learn_inducing_points: bool = False,
        base_covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
        **kwargs,
    ) -> None:
        self._num_outputs = num_outputs
        self._input_batch_shape = train_X.shape[:-2]
        aug_batch_shape = copy.deepcopy(self._input_batch_shape)
        if num_outputs > 1:
            # I don't understand what mypy wants here
            aug_batch_shape += torch.Size([num_outputs])  # type: ignore
        self._aug_batch_shape = aug_batch_shape

        if likelihood is None:
            likelihood = self.default_likelihood

        if task_dims is None:
            task_dims = [0]

        if num_tasks is None:
            num_tasks = [1 for _ in task_dims]

        if ranks is None:
            ranks = [1 for _ in task_dims]

        if base_covar_module is None:
            base_covar_module = MaternKernel(
                nu=2.5,
                ard_num_dims=train_X.shape[-1],
                batch_shape=self._aug_batch_shape,
                lengthscale_prior=GammaPrior(3.0, 6.0),
            ).to(train_X)

        index_modules = []
        for task_dim, num_task, rank in zip(task_dims, num_tasks, ranks):
            index_module = gpytorch.kernels.IndexKernel(
                num_tasks=num_task,
                rank=rank,
                active_dims=task_dim,
                ard_num_dims=1,
                prior=gpytorch.priors.LKJCovariancePrior(
                    n=num_task,
                    eta=1.5,
                    sd_prior=gpytorch.priors.GammaPrior(1.0, 0.15),
                ),
            )
            index_modules.append(index_module)
        covar_module = ProductKernel(base_covar_module, *index_modules)

        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            num_outputs=num_outputs,
            learn_inducing_points=learn_inducing_points,
            covar_module=covar_module,
            mean_module=mean_module,
            variational_distribution=variational_distribution,
            variational_strategy=variational_strategy,
            inducing_points=inducing_points,
            outcome_transform=outcome_transform,
            input_transform=input_transform,
            inducing_point_allocator=inducing_point_allocator,
            **kwargs,
        )


class BetaRegressionGP(VariationalGP):
    outcome_type = "percentage"
    default_likelihood = BetaLikelihood()


class OrdinalGP(VariationalGP):
    """
    Convenience class for using a VariationalGP with an OrdinalLikelihood.
    """

    outcome_type = "ordinal"
    default_likelihood = OrdinalLikelihood(n_levels=3)

    def predict_probability(self, x: Union[torch.Tensor, np.ndarray]):
        fmean, fvar = super().predict(x)
        return OrdinalGPModel.calculate_probs(self, fmean, fvar)

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None):
        options = super().get_config_options(config)

        dim = get_dim(config)

        if config.getobj(cls.__name__, "mean_covar_factory", fallback=None) is None:
            mean, covar = ordinal_mean_covar_factory(config)
            options["mean_covar_factory"] = (mean, covar)
            ls_prior = gpytorch.priors.GammaPrior(concentration=1.5, rate=3.0)
            ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate
            ls_constraint = gpytorch.constraints.Positive(
                transform=None, initial_value=ls_prior_mode
            )

            # no outputscale due to shift identifiability in d.
            covar_module = gpytorch.kernels.RBFKernel(
                lengthscale_prior=ls_prior,
                lengthscale_constraint=ls_constraint,
                ard_num_dims=dim,
            )

            options["covar_module"] = covar_module

        assert options["inducing_size"] >= 1, "Inducing size must be non-zero."

        return options
