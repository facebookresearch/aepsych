#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

import gpytorch
import torch
from aepsych.acquisition.objective import FloorLogitObjective
from aepsych.acquisition.objective.semi_p import SemiPThresholdObjective
from aepsych.config import Config
from aepsych.likelihoods import BernoulliObjectiveLikelihood, LinearBernoulliLikelihood
from aepsych.models import GPClassificationModel
from aepsych.models.inducing_points import GreedyVarianceReduction
from aepsych.models.inducing_points.base import InducingPointAllocator
from aepsych.utils import get_dims, get_optimizer_options, promote_0d
from aepsych.utils_logging import getLogger
from botorch.acquisition.objective import PosteriorTransform
from botorch.optim.fit import fit_gpytorch_mll_scipy
from botorch.posteriors import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import BernoulliLikelihood, Likelihood
from gpytorch.means import ConstantMean, ZeroMean
from gpytorch.priors import GammaPrior
from torch.distributions import Normal

# TODO: Implement a covar factory and analytic method for getting the lse
logger = getLogger()


def _hadamard_mvn_approx(
    x_intensity: torch.Tensor,
    slope_mean: torch.Tensor,
    slope_cov: torch.Tensor,
    offset_mean: torch.Tensor,
    offset_cov: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    MVN approximation to the hadamard product of GPs (from the SemiP paper, extending the
    zero-mean results in https://mathoverflow.net/questions/293955/normal-approximation-to-the-pointwise-hadamard-schur-product-of-two-multivariat)

    Args:
        x_intensity (torch.Tensor): The intensity dimension
        slope_mean (torch.Tensor): The mean of the slope GP
        slope_cov (torch.Tensor): The covariance of the slope GP
        offset_mean (torch.Tensor): The mean of the offset GP
        offset_cov (torch.Tensor): The covariance of the offset GP

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The mean and covariance of the approximated MVN
    """
    offset_mean = offset_mean + x_intensity

    mean_x = offset_mean * slope_mean

    # Same as torch.diag_embed(slope_mean) @ offset_cov @ torch.diag_embed(slope_mean), but more efficient
    term1 = slope_mean.unsqueeze(-1) * offset_cov * slope_mean.unsqueeze(-2)

    # Same as torch.diag_embed(offset_mean) @ slope_cov @ torch.diag_embed(offset_mean), but more efficient
    term2 = offset_mean.unsqueeze(-1) * slope_cov * offset_mean.unsqueeze(-2)

    term3 = slope_cov * offset_cov

    cov_x = term1 + term2 + term3

    return mean_x, cov_x


def semi_p_posterior_transform(posterior: GPyTorchPosterior) -> GPyTorchPosterior:
    """Transform a posterior from a SemiP model to a Hadamard model.

    Args:
        posterior (GPyTorchPosterior): The posterior to transform

    Returns:
        GPyTorchPosterior: The transformed posterior.
    """
    batch_mean = posterior.mvn.mean
    batch_cov = posterior.mvn.covariance_matrix
    offset_mean = batch_mean[..., 0, :]
    slope_mean = batch_mean[..., 1, :]
    offset_cov = batch_cov[..., 0, :, :]
    slope_cov = batch_cov[..., 1, :, :]
    Xi = posterior.Xi
    approx_mean, approx_cov = _hadamard_mvn_approx(
        x_intensity=Xi,
        slope_mean=slope_mean,
        slope_cov=slope_cov,
        offset_mean=offset_mean,
        offset_cov=offset_cov,
    )
    approx_mvn = MultivariateNormal(mean=approx_mean, covariance_matrix=approx_cov)
    return GPyTorchPosterior(distribution=approx_mvn)


class SemiPPosterior(GPyTorchPosterior):
    def __init__(
        self,
        mvn: MultivariateNormal,
        likelihood: LinearBernoulliLikelihood,
        Xi: torch.Tensor,
    ) -> None:
        """Initialize a SemiPPosterior object.

        Args:
            mvn (MultivariateNormal): The MVN object to use
            likelihood (LinearBernoulliLikelihood): The likelihood object
            Xi (torch.Tensor): The intensity dimension
        """

        super().__init__(distribution=mvn)
        self.likelihood = likelihood
        self.Xi = Xi

    def rsample_from_base_samples(
        self,
        sample_shape: torch.Size,
        base_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from the posterior (with gradients) using base samples.

        This is intended to be used with a sampler that produces the corresponding base
        samples, and enables acquisition optimization via Sample Average Approximation.

        Args:
            sample_shape (torch.Size): The desired shape of the samples
            base_samples (torch.Tensor): The base samples

        Returns:
            torch.Tensor: The sampled values from the posterior distribution
        """
        return (
            super()
            .rsample_from_base_samples(
                sample_shape=sample_shape, base_samples=base_samples
            )
            .squeeze(-1)
        )

    def rsample(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the posterior distribution using the reparameterization trick

        Args:
            sample_shape (torch.Size, optional): The desired shape of the samples. Defaults to None.
            base_samples (torch.Tensor, optional): The base samples. Defaults to None.

        Returns:
            torch.Tensor: The sampled values from the posterior distribution.
        """
        if base_samples is None:
            samps_ = super().rsample(sample_shape=sample_shape)
        else:
            samps_ = super().rsample_from_base_samples(
                sample_shape=sample_shape,
                base_samples=base_samples.expand(
                    self._extended_shape(sample_shape)
                ).squeeze(-1),
            )
        kcsamps = samps_.squeeze(-1)
        # fsamps is of shape nsamp x 2 x n, or nsamp x b x 2 x n
        return kcsamps

    def sample_p(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the likelihood distribution of the modeled function.

        Args:
            sample_shape (torch.Size, optional): The desired shape of the samples. Defaults to None.
            base_samples (torch.Tensor, optional): The base samples. Defaults to None.

        Returns:
            torch.Tensor: The sampled values from the likelihood distribution.
        """
        kcsamps = self.rsample(sample_shape=sample_shape, base_samples=base_samples)
        return self.likelihood.p(function_samples=kcsamps, Xi=self.Xi).squeeze(-1)

    def sample_f(
        self,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample from the function values of the modeled distribution.

        Args:
            sample_shape (torch.Size, optional): The desired shape of the samples. Defaults to None.
            base_samples (torch.Tensor, optional): The base samples. Defaults to None.

        Returns:
            torch.Tensor: The sampled function values from the likelihood.
        """

        kcsamps = self.rsample(sample_shape=sample_shape, base_samples=base_samples)
        return self.likelihood.f(function_samples=kcsamps, Xi=self.Xi).squeeze(-1)

    def sample_thresholds(
        self,
        threshold_level: float,
        sample_shape: Optional[torch.Size] = None,
        base_samples: Optional[torch.Tensor] = None,
    ) -> SemiPThresholdObjective:
        """Sample the thresholds based on the given threshold level.

        Args:
        threshold_level (float): The target threshold level for sampling.
        sample_shape (torch.Size, optional): The desired shape of the samples. Defaults to None.
        base_samples (torch.Tensor, optional): The base samples. Defaults to None.

        Returns:
            SemiPThresholdObjective: The sampled thresholds based on the threshold level.
        """

        fsamps = self.rsample(sample_shape=sample_shape, base_samples=base_samples)
        return SemiPThresholdObjective(
            likelihood=self.likelihood, target=threshold_level
        )(samples=fsamps, X=self.Xi)


class SemiParametricGPModel(GPClassificationModel):
    """
    Semiparametric GP model for psychophysics.

    Implements a semi-parametric model with a functional form like :math:`k(x_c()x_i + c(x_c))`,
    for scalar intensity dimension :math:`x_i` and vector-valued context dimensions :math:`x_c`,
    with k and c having a GP prior. In contrast to HadamardSemiPModel, this version uses a batched GP
    directly, which is about 2-3x slower but does not use the MVN approximation.

    Intended for use with a BernoulliObjectiveLikelihood with flexible link function such as
    Logistic or Gumbel nonlinearity with a floor.
    """

    _num_outputs = 1
    _batch_shape = 2
    stimuli_per_trial = 1
    outcome_type = "binary"

    def __init__(
        self,
        dim: int,
        stim_dim: int = 0,
        mean_module: Optional[gpytorch.means.Mean] = None,
        covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Any] = None,
        slope_mean: float = 2,
        inducing_point_method: Optional[InducingPointAllocator] = None,
        inducing_size: int = 100,
        max_fit_time: Optional[float] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize SemiParametricGP.
        Args:
            dim (int, optional): The number of dimensions in the parameter space.
            stim_dim (int): Index of the intensity (monotonic) dimension. Defaults to 0.
            mean_module (gpytorch.means.Mean, optional): GP mean class. Defaults to a constant with a normal prior.
            covar_module (gpytorch.kernels.Kernel, optional): GP covariance kernel class. Defaults to scaled RBF with a
                gamma prior.
            likelihood (gpytorch.likelihood.Likelihood, optional): The likelihood function to use. If None defaults to
                linear-Bernouli likelihood with probit link.
            slope_mean (float): The mean of the slope. Defaults to 2.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (Dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """

        self.dim = dim
        self.stim_dim = stim_dim
        self.context_dims = list(range(self.dim))
        self.context_dims.pop(stim_dim)

        if mean_module is None:
            mean_module = ConstantMean(batch_shape=torch.Size([2]))
            mean_module.requires_grad_(False)
            mean_module.constant.copy_(
                torch.tensor([0.0, slope_mean])  # offset mean is 0, slope mean is 2
            )

        if covar_module is None:
            covar_module = ScaleKernel(
                RBFKernel(
                    ard_num_dims=self.dim - 1,
                    lengthscale_prior=GammaPrior(3, 6),
                    active_dims=self.context_dims,  # Operate only on x_s
                    batch_shape=torch.Size([2]),
                ),
                outputscale_prior=GammaPrior(1.5, 1.0),
            )

        likelihood = likelihood or LinearBernoulliLikelihood()
        assert isinstance(
            likelihood, LinearBernoulliLikelihood
        ), "SemiP model only supports linear Bernoulli likelihoods!"

        super().__init__(
            dim=dim,
            mean_module=mean_module,
            covar_module=covar_module,
            likelihood=likelihood,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            optimizer_options=optimizer_options,
        )

    @classmethod
    def from_config(cls, config: Config) -> SemiParametricGPModel:
        """Alternate constructor for SemiParametricGPModel model.

        This is used when we recursively build a full sampling strategy
        from a configuration.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            SemiParametricGPModel: Configured class instance.
        """

        classname = cls.__name__
        inducing_size = config.getint(classname, "inducing_size", fallback=100)

        dim = config.getint(classname, "dim", fallback=None)

        if dim is None:
            dim = get_dims(config)

        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        inducing_point_method_class = config.getobj(
            classname, "inducing_point_method", fallback=GreedyVarianceReduction
        )
        # Check if allocator class has a `from_config` method
        if hasattr(inducing_point_method_class, "from_config"):
            inducing_point_method = inducing_point_method_class.from_config(config)
        else:
            inducing_point_method = inducing_point_method_class()
        likelihood_cls = config.getobj(classname, "likelihood", fallback=None)

        if hasattr(likelihood_cls, "from_config"):
            likelihood = likelihood_cls.from_config(config)
        elif likelihood_cls is not None:
            likelihood = likelihood_cls()
        else:
            likelihood = None

        stim_dim = config.getint(classname, "stim_dim", fallback=0)

        slope_mean = config.getfloat(classname, "slope_mean", fallback=2)

        optimizer_options = get_optimizer_options(config, classname)

        return cls(
            stim_dim=stim_dim,
            dim=dim,
            likelihood=likelihood,
            slope_mean=slope_mean,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            optimizer_options=optimizer_options,
        )

    def fit(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        warmstart_hyperparams: bool = False,
        warmstart_induc: bool = False,
        **kwargs,
    ) -> None:
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
            warmstart_hyperparams (bool): Whether to reuse the previous hyperparameters (True) or fit from scratch
                (False). Defaults to False.
            warmstart_induc (bool): Whether to reuse the previous inducing points or fit from scratch (False).
                Defaults to False.
            kwargs: Keyword arguments passed to `optimizer=fit_gpytorch_mll_scipy`.
        """
        super().fit(
            train_x=train_x,
            train_y=train_y,
            optimizer=fit_gpytorch_mll_scipy,
            warmstart_hyperparams=warmstart_hyperparams,
            warmstart_induc=warmstart_induc,
            closure_kwargs={"Xi": train_x[..., self.stim_dim]},
            **kwargs,
        )

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int,
        probability_space: bool = False,
    ) -> torch.Tensor:
        """Sample from underlying model.

        Args:

            x (torch.Tensor): `n x d` Points at which to sample.
            num_samples (int): Number of samples to return. Defaults to None.
            probability_space (bool): Whether to sample from the probability space (True) or the latent function. Defaults to False.
            kwargs are ignored

        Returns:
            (num_samples x n) torch.Tensor: Posterior samples
        """
        post = self.posterior(x)
        if probability_space is True:
            samps = post.sample_p(torch.Size([num_samples])).detach()
        else:
            samps = post.sample_f(torch.Size([num_samples])).detach()

        assert samps.shape == (num_samples, 1, x.shape[0])
        return samps.squeeze(1)

    def predict(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at query points.
        """
        with torch.no_grad():
            samps = self.sample(
                x, num_samples=10000, probability_space=probability_space
            )
        m, v = samps.mean(0), samps.var(0)
        return promote_0d(m), promote_0d(v)

    def posterior(
        self, X: torch.Tensor, posterior_transform: Optional[PosteriorTransform] = None
    ) -> SemiPPosterior:
        """Get the posterior distribution at the given points.

        Args:
            X (torch.Tensor): Points at which to evaluate the posterior.
            posterior_transform (PosteriorTransform, optional): A transform to apply to the posterior. Defaults to None.

        Returns:
            SemiPPosterior: The posterior distribution at the given points.
        """
        # Assume x is (b) x n x d
        if X.ndim > 3:
            raise ValueError
        # Add in the extra 2 batch for the 2 GPs in this model
        Xnew = X.unsqueeze(
            -3
        ).expand(
            X.shape[:-2]  # (b)
            + torch.Size([2])  # For the two GPs
            + X.shape[-2:]  # n x d
        )
        # The shape of Xnew is: (b) x 2 x n x d
        posterior = SemiPPosterior(
            mvn=self(Xnew),
            likelihood=self.likelihood,
            Xi=X[..., self.stim_dim],
        )

        if posterior_transform is not None:
            return posterior_transform(posterior)
        else:
            return posterior


class HadamardSemiPModel(GPClassificationModel):
    """
    Semiparametric GP model for psychophysics, with a MVN approximation to the elementwise
    product of GPs.

    Implements a semi-parametric model with a functional form like :math:`k(x_c()x_i + c(x_c))`,
    for scalar intensity dimension :math:`x_i` and vector-valued context dimensions :math:`x_c`,
    with k and c having a GP prior. In contrast to SemiParametricGPModel, this version approximates
    the product as a single multivariate normal, which should be faster (the approximation is exact
    if one of the GP's variance goes to zero).
    Intended for use with a BernoulliObjectiveLikelihood with flexible link function such as
    Logistic or Gumbel nonlinearity with a floor.
    """

    _num_outputs = 1
    stimuli_per_trial = 1
    outcome_type = "binary"

    def __init__(
        self,
        dim: int,
        stim_dim: int = 0,
        slope_mean_module: Optional[gpytorch.means.Mean] = None,
        slope_covar_module: Optional[gpytorch.kernels.Kernel] = None,
        offset_mean_module: Optional[gpytorch.means.Mean] = None,
        offset_covar_module: Optional[gpytorch.kernels.Kernel] = None,
        likelihood: Optional[Likelihood] = None,
        slope_mean: float = 2,
        inducing_point_method: Optional[InducingPointAllocator] = None,
        inducing_size: int = 100,
        max_fit_time: Optional[float] = None,
        optimizer_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize HadamardSemiPModel.
        Args:
            dim (int): The number of dimensions in the parameter space.
            stim_dim (int): Index of the intensity (monotonic) dimension. Defaults to 0.
            slope_mean_module (gpytorch.means.Mean, optional): Mean module to use (default: constant mean) for slope.
            slope_covar_module (gpytorch.kernels.Kernel, optional): Covariance kernel to use (default: scaled RBF) for slope.
            offset_mean_module (gpytorch.means.Mean, optional): Mean module to use (default: constant mean) for offset.
            offset_covar_module (gpytorch.kernels.Kernel, optional): Covariance kernel to use (default: scaled RBF) for offset.
            likelihood (gpytorch.likelihood.Likelihood, optional)): defaults to bernoulli with logistic input and a floor of .5
            slope_mean (float): The mean of the slope. Defaults to 2.
            inducing_point_method (InducingPointAllocator, optional): The method to use for selecting inducing points.
                If not set, a GreedyVarianceReduction is made.
            inducing_size (int): Number of inducing points. Defaults to 100.
            max_fit_time (float, optional): The maximum amount of time, in seconds, to spend fitting the model. If None,
                there is no limit to the fitting time.
            optimizer_options (Dict[str, Any], optional): Optimizer options to pass to the SciPy optimizer during
                fitting. Assumes we are using L-BFGS-B.
        """
        super().__init__(
            dim=dim,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            optimizer_options=optimizer_options,
        )

        self.stim_dim = stim_dim

        if slope_mean_module is None:
            self.slope_mean_module = ConstantMean()
            self.slope_mean_module.requires_grad_(False)
            self.slope_mean_module.constant.copy_(
                torch.tensor(slope_mean)
            )  # magic number to shift the slope prior to be generally positive.
        else:
            self.slope_mean_module = slope_mean_module

        if offset_mean_module is None:
            self.offset_mean_module = ZeroMean()
        else:
            self.offset_mean_module = offset_mean_module

        self.offset_mean_module = offset_mean_module or ZeroMean()

        context_dims = list(range(self.dim))
        context_dims.pop(stim_dim)

        self.slope_covar_module = slope_covar_module or ScaleKernel(
            RBFKernel(
                ard_num_dims=self.dim - 1,
                lengthscale_prior=GammaPrior(3, 6),
                active_dims=context_dims,  # Operate only on x_s
            ),
            outputscale_prior=GammaPrior(1.5, 1.0),
        )

        self.offset_covar_module = offset_covar_module or ScaleKernel(
            RBFKernel(
                ard_num_dims=self.dim - 1,
                lengthscale_prior=GammaPrior(3, 6),
                active_dims=context_dims,  # Operate only on x_s
            ),
            outputscale_prior=GammaPrior(1.5, 1.0),
        )
        self.likelihood = likelihood or BernoulliObjectiveLikelihood(
            objective=FloorLogitObjective()
        )

        self._fresh_state_dict = deepcopy(self.state_dict())
        self._fresh_likelihood_dict = deepcopy(self.likelihood.state_dict())

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        """Forward pass for HadamardSemiPModel GP.

        generates a k(c + x[:,stim_dim]) = kc + kx[:,stim_dim] mvn object where k and c are
        slope and offset GPs and x[:,stim_dim] are the intensity stimulus (x)
        locations and thus acts as a constant offset to the k mvn.
        Args:
            x (torch.Tensor): Points at which to sample.

        Returns:
            MVN object evaluated at samples
        """
        # TODO: make slope prop to intensity width.
        slope_mean = self.slope_mean_module(x)

        # kc mvn
        offset_mean = self.offset_mean_module(x)

        slope_cov = self.slope_covar_module(x)
        offset_cov = self.offset_covar_module(x)

        mean_x, cov_x = _hadamard_mvn_approx(
            x_intensity=x[..., self.stim_dim],
            slope_mean=slope_mean,
            slope_cov=slope_cov,
            offset_mean=offset_mean,
            offset_cov=offset_cov,
        )
        return MultivariateNormal(mean_x, cov_x)

    @classmethod
    def from_config(cls, config: Config) -> HadamardSemiPModel:
        """Alternate constructor for HadamardSemiPModel model.

        This is used when we recursively build a full sampling strategy
        from a configuration.

        Args:
            config (Config): A configuration containing keys/values matching this class

        Returns:
            HadamardSemiPModel: Configured class instance.
        """

        classname = cls.__name__
        inducing_size = config.getint(classname, "inducing_size", fallback=100)

        dim = config.getint(classname, "dim", fallback=None)
        if dim is None:
            dim = get_dims(config)

        slope_mean_module = config.getobj(classname, "slope_mean_module", fallback=None)
        slope_covar_module = config.getobj(
            classname, "slope_covar_module", fallback=None
        )
        offset_mean_module = config.getobj(
            classname, "offset_mean_module", fallback=None
        )
        offset_covar_module = config.getobj(
            classname, "offset_covar_module", fallback=None
        )

        max_fit_time = config.getfloat(classname, "max_fit_time", fallback=None)

        inducing_point_method_class = config.getobj(
            classname, "inducing_point_method", fallback=GreedyVarianceReduction
        )
        # Check if allocator class has a `from_config` method
        if hasattr(inducing_point_method_class, "from_config"):
            inducing_point_method = inducing_point_method_class.from_config(config)
        else:
            inducing_point_method = inducing_point_method_class()
        likelihood_cls = config.getobj(classname, "likelihood", fallback=None)
        if hasattr(likelihood_cls, "from_config"):
            likelihood = likelihood_cls.from_config(config)
        else:
            likelihood = likelihood_cls()

        slope_mean = config.getfloat(classname, "slope_mean", fallback=2)

        stim_dim = config.getint(classname, "stim_dim", fallback=0)

        optimizer_options = get_optimizer_options(config, classname)

        return cls(
            stim_dim=stim_dim,
            dim=dim,
            slope_mean_module=slope_mean_module,
            slope_covar_module=slope_covar_module,
            offset_mean_module=offset_mean_module,
            offset_covar_module=offset_covar_module,
            likelihood=likelihood,
            slope_mean=slope_mean,
            inducing_size=inducing_size,
            max_fit_time=max_fit_time,
            inducing_point_method=inducing_point_method,
            optimizer_options=optimizer_options,
        )

    def predict(
        self, x: torch.Tensor, probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at queries points.
        """
        if probability_space:
            if hasattr(self.likelihood, "objective"):
                fsamps = self.sample(x, 1000)
                psamps = self.likelihood.objective(fsamps)
                return psamps.mean(0).squeeze(), psamps.var(0).squeeze()
            elif isinstance(self.likelihood, BernoulliLikelihood):  # default to probit
                fsamps = self.sample(x, 1000)
                psamps = Normal(0, 1).cdf(fsamps)
                return psamps.mean(0).squeeze(), psamps.var(0).squeeze()
            else:
                raise NotImplementedError(
                    f"p-space sampling not defined if likelihood ({self.likelihood}) does not have a link!"
                )
        else:
            with torch.no_grad():
                post = self.posterior(x)
            fmean = post.mean.squeeze()
            fvar = post.variance.squeeze()
            return fmean, fvar
