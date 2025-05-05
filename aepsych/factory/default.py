#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import math
import warnings
from configparser import NoOptionError
from copy import deepcopy
from typing import Any, Literal

import gpytorch
import torch
from aepsych.config import Config, ConfigurableMixin
from aepsych.utils import get_dims
from scipy.stats import norm

from .utils import (
    __default_invgamma_concentration,
    __default_invgamma_rate,
    DEFAULT_INVGAMMA_CONC,
    DEFAULT_INVGAMMA_RATE,
)

# The gamma lengthscale prior is taken from
# https://betanalpha.github.io/assets/case_studies/gaussian_processes.html#323_Informative_Prior_Model

# The lognormal lengscale prior is taken from
# https://arxiv.org/html/2402.02229v3


class MeanCovarFactory(ConfigurableMixin, abc.ABC):
    def __init__(self, dim: int, stimuli_per_trial: int = 1, *args, **kwargs) -> None:
        """Abstract base class for mean and covariance function factories.

        Args:
            dim (int): Dimensionality of the parameter space.
            stimuli_per_trial (int, optional): Number of stimuli per trial. Defaults to 1.
        """
        self.dim = dim
        self.stimuli_per_trial = stimuli_per_trial

        self.mean_module = self._make_mean_module()
        self.covar_module = self._make_covar_module()

    def get_mean(self) -> gpytorch.means.Mean:
        return deepcopy(self.mean_module)

    def get_covar(self) -> gpytorch.kernels.Kernel:
        return deepcopy(self.covar_module)

    @abc.abstractmethod
    def _make_mean_module(self) -> gpytorch.means.Mean:
        pass

    @abc.abstractmethod
    def _make_covar_module(self) -> gpytorch.kernels.Kernel:
        pass

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the MeanCovarFactory.

        Args:
            config (Config): Config object to find options in.
            name (str, optional): Name of the factory. Defaults to the class name.
            options (dict, optional): Options to start with. Defaults to None.

        Returns:
            dict[str, Any]: Options to use to initialize the factory.
        """
        name = name or cls.__name__
        options = super().get_config_options(config, name, options)

        if "dim" not in options:
            options["dim"] = get_dims(config)

        return options


class DefaultMeanCovarFactory(MeanCovarFactory):
    def __init__(
        self,
        dim: int,
        stimuli_per_trial: int = 1,
        zero_mean: bool = False,
        target: float | None = None,
        cov_kernel: gpytorch.kernels.Kernel = gpytorch.kernels.RBFKernel,
        active_dims: list[int] | None = None,
        lengthscale_prior: Literal["invgamma", "gamma", "lognormal"] | None = None,
        ls_loc: torch.Tensor | float | None = None,
        ls_scale: torch.Tensor | float | None = None,
        fixed_kernel_amplitude: bool | None = None,
        outputscale_prior: Literal["box", "gamma"] = "box",
    ) -> None:
        """Factory that makes mean and covariance functions for generic GPs.
        After initialization, copies of the mean and covariance functions can be made with
        `get_mean` and `get_covar`.

        Args:
            dim (int, optional): Dimensionality of the parameter space. Must be provided.
            stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.
            zero_mean (bool, optional): Whether to use zero for the mean module. Defaults to False.
            target (float, optional): Target for the mean module. Defaults to None.
            cov_kernel (gpytorch.kernels.Kernel, optional): Covariance kernel to use. Defaults to RBF
                kernel.
            active_dims (list[int], optional): List of dimensions to use in the covariance function. Defaults to None,
                which uses all dimensions.
            lengthscale_prior (Literal["invgamma", "gamma", "lognormal"], optional): Prior to use for
                lengthscale. Defaults to "lognormal" if stimuli_per_trial == 1, else "gamma".
            ls_loc (torch.Tensor | float, optional): Location parameter for lengthscale prior.
                Defaults to sqrt(2.0).
            ls_scale (torch.Tensor | float, optional): Scale parameter for lengthscale prior.
                Defaults to sqrt(3.0).
            fixed_kernel_amplitude (bool, optional): Whether to allow the covariance kernel to scale.
                Defaults to True if stimuli_per_trial == 1, else False.
            outputscale_prior (Literal["box", "gamma"], optional): Prior to use to scale the covariance kernel.
                Defaults to "box".
        """
        self.zero_mean = zero_mean
        self.target = target
        self.cov_kernel = cov_kernel
        self.active_dims = active_dims
        self.lengthscale_prior = lengthscale_prior
        self.ls_loc = ls_loc
        self.ls_scale = ls_scale
        self.fixed_kernel_amplitude = fixed_kernel_amplitude
        self.outputscale_prior = outputscale_prior

        super().__init__(dim, stimuli_per_trial)

    def get_mean(self) -> gpytorch.means.Mean:
        return deepcopy(self.mean_module)

    def get_covar(self) -> gpytorch.kernels.Kernel:
        return deepcopy(self.covar_module)

    def _make_mean_module(self) -> gpytorch.means.Mean:
        # Make mean module
        if self.zero_mean:
            mean = gpytorch.means.ZeroMean()
        else:
            mean = gpytorch.means.ConstantMean()

        if self.target is not None:
            if self.zero_mean:
                warnings.warn(
                    "Specified both `zero_mean = True` and `target`. Zero mean will be overwritten by target fixed mean!",
                    UserWarning,
                    stacklevel=2,
                )

            mean.constant.requires_grad_(False)
            mean.constant.copy_(torch.tensor(norm.ppf(self.target)))

        return mean

    def _make_covar_module(self) -> gpytorch.kernels.Kernel:
        # Make covariance module
        if self.ls_loc is None:
            self.ls_loc = torch.tensor(math.sqrt(2.0), dtype=torch.float64)
        elif not isinstance(self.ls_loc, torch.Tensor):
            self.ls_loc = torch.tensor(self.ls_loc, dtype=torch.float64)

        if self.ls_scale is None:
            self.ls_scale = torch.tensor(math.sqrt(3.0), dtype=torch.float64)
        elif not isinstance(self.ls_scale, torch.Tensor):
            self.ls_scale = torch.tensor(self.ls_scale, dtype=torch.float64)

        if self.fixed_kernel_amplitude is None:
            self.fixed_kernel_amplitude = True if self.stimuli_per_trial == 1 else False

        if self.lengthscale_prior == "invgamma":
            ls_prior = gpytorch.priors.GammaPrior(
                concentration=DEFAULT_INVGAMMA_CONC,
                rate=DEFAULT_INVGAMMA_RATE,
                transform=lambda x: 1 / x,
            )
            ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)

        elif self.lengthscale_prior == "gamma" or (
            self.lengthscale_prior is None and self.stimuli_per_trial != 1
        ):
            ls_prior = gpytorch.priors.GammaPrior(concentration=3.0, rate=6.0)
            ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate

        elif self.lengthscale_prior == "lognormal" or (
            self.lengthscale_prior is None and self.stimuli_per_trial == 1
        ):
            ls_prior = gpytorch.priors.LogNormalPrior(
                self.ls_loc + math.log(self.dim) / 2, self.ls_scale
            )
            ls_prior_mode = torch.exp(self.ls_loc - self.ls_scale**2)
        else:
            raise RuntimeError(
                f"Lengthscale_prior should be invgamma, gamma, or lognormal, got {self.lengthscale_prior}"
            )

        ls_constraint = gpytorch.constraints.GreaterThan(
            lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
        )

        covar = self.cov_kernel(
            lengthscale_prior=ls_prior,
            lengthscale_constraint=ls_constraint,
            ard_num_dims=self.dim,
            active_dims=self.active_dims,
        )
        if not self.fixed_kernel_amplitude:
            if self.outputscale_prior == "gamma":
                os_prior = gpytorch.priors.GammaPrior(concentration=2.0, rate=0.15)
            elif self.outputscale_prior == "box":
                os_prior = gpytorch.priors.SmoothedBoxPrior(a=1, b=4)
            else:
                raise RuntimeError(
                    f"Outputscale_prior should be gamma or box, got {self.outputscale_prior}"
                )

            covar = gpytorch.kernels.ScaleKernel(
                covar,
                outputscale_prior=os_prior,
                outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
            )

        return covar

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get configuration options for the MeanCovarFactory.

        Args:
            config (Config): Config object to find options in.
            name (str, optional): Name of the factory. Defaults to the class name.
            options (dict, optional): Options to start with. Defaults to None.

        Returns:
            dict[str, Any]: Options to use to initialize the factory.
        """
        name = name or cls.__name__
        options = super().get_config_options(config, name, options)

        if "dim" not in options:
            options["dim"] = get_dims(config)

        return options


def default_mean_covar_factory(
    config: Config | None = None,
    dim: int | None = None,
    stimuli_per_trial: int = 1,
) -> tuple[gpytorch.means.ConstantMean, gpytorch.kernels.ScaleKernel]:
    """Default factory for generic GP models

    Args:
        config (Config, optional): Object containing bounds (and potentially other
            config details).
        dim (int, optional): Dimensionality of the parameter space. Must be provided
            if config is None.
        stimuli_per_trial (int): Number of stimuli per trial. Defaults to 1.

    Returns:
        tuple[gpytorch.means.Mean, gpytorch.kernels.Kernel]: Instantiated
            ConstantMean and ScaleKernel with priors based on bounds.
    """
    warnings.warn(
        "default_mean_covar_factory is deprecated, use the DefaultMeanCovarFactory class instead!",
        DeprecationWarning,
        stacklevel=2,
    )

    assert (config is not None) or (
        dim is not None
    ), "Either config or dim must be provided!"

    assert stimuli_per_trial in (1, 2), "stimuli_per_trial must be 1 or 2!"

    zero_mean = False
    if config is not None:
        lb = config.gettensor("default_mean_covar_factory", "lb")
        ub = config.gettensor("default_mean_covar_factory", "ub")
        assert lb.shape[0] == ub.shape[0], "bounds shape mismatch!"
        config_dim: int = lb.shape[0]

        if dim is not None:
            assert dim == config_dim, "Provided config does not match provided dim!"
        else:
            dim = config_dim

        zero_mean = config.getboolean(
            "default_mean_covar_factory", "zero_mean", fallback=False
        )

    mean = _get_default_mean_function(config, zero_mean)
    covar = _get_default_cov_function(config, dim, stimuli_per_trial)  # type: ignore

    return mean, covar


def _get_default_mean_function(
    config: Config | None = None, zero_mean: bool = False
) -> gpytorch.means.ConstantMean:
    """Creates a default mean function for Gaussian Processes.

    Args:
        config (Config, optional): Configuration object.

    Returns:
        gpytorch.means.ConstantMean: An instantiated mean function with appropriate priors based on the configuration.
    """
    if zero_mean:
        mean = gpytorch.means.ZeroMean()
    else:
        mean = gpytorch.means.ConstantMean()

    if config is not None:
        fixed_mean = config.getboolean(
            "default_mean_covar_factory", "fixed_mean", fallback=False
        )
        if fixed_mean:
            if zero_mean:
                warnings.warn(
                    "Specified both `zero_mean = True` and `fixed_mean = True`. Deferring to fixed_mean!"
                )
            try:
                target = config.getfloat("default_mean_covar_factory", "target")
                mean.constant.requires_grad_(False)
                mean.constant.copy_(torch.tensor(norm.ppf(target)))
            except NoOptionError:
                raise RuntimeError("Config got fixed_mean=True but no target included!")

    return mean


def _get_default_cov_function(
    config: Config | None,
    dim: int,
    stimuli_per_trial: int,
    active_dims: list[int] | None = None,
) -> gpytorch.kernels.Kernel:
    """Creates a default covariance function for Gaussian Processes.

    Args:
        config (Config, optional): Configuration object.
        dim (int): Dimensionality of the parameter space.
        stimuli_per_trial (int): Number of stimuli per trial.
        active_dims (list[int], optional): List of dimensions to use in the covariance function. Defaults to None.

    Returns:
        gpytorch.kernels.Kernel: An instantiated kernel with appropriate priors based on the configuration.
    """

    # default priors
    lengthscale_prior = "lognormal" if stimuli_per_trial == 1 else "gamma"
    ls_loc = torch.tensor(math.sqrt(2.0), dtype=torch.float64)
    ls_scale = torch.tensor(math.sqrt(3.0), dtype=torch.float64)
    fixed_kernel_amplitude = True if stimuli_per_trial == 1 else False
    outputscale_prior = "box"
    kernel = gpytorch.kernels.RBFKernel

    if config is not None:
        lengthscale_prior = config.get(
            "default_mean_covar_factory",
            "lengthscale_prior",
            fallback=lengthscale_prior,
        )
        if lengthscale_prior == "lognormal":
            ls_loc = config.gettensor(
                "default_mean_covar_factory",
                "ls_loc",
                fallback=ls_loc,
            )
            ls_scale = config.gettensor(
                "default_mean_covar_factory", "ls_scale", fallback=ls_scale
            )
        fixed_kernel_amplitude = config.getboolean(
            "default_mean_covar_factory",
            "fixed_kernel_amplitude",
            fallback=fixed_kernel_amplitude,
        )
        outputscale_prior = config.get(
            "default_mean_covar_factory",
            "outputscale_prior",
            fallback=outputscale_prior,
        )

        kernel = config.getobj("default_mean_covar_factory", "kernel", fallback=kernel)

    if lengthscale_prior == "invgamma":
        ls_prior = gpytorch.priors.GammaPrior(
            concentration=__default_invgamma_concentration,
            rate=__default_invgamma_rate,
            transform=lambda x: 1 / x,
        )
        ls_prior_mode = ls_prior.rate / (ls_prior.concentration + 1)

    elif lengthscale_prior == "gamma":
        ls_prior = gpytorch.priors.GammaPrior(concentration=3.0, rate=6.0)
        ls_prior_mode = (ls_prior.concentration - 1) / ls_prior.rate

    elif lengthscale_prior == "lognormal":
        if not isinstance(ls_loc, torch.Tensor):
            ls_loc = torch.tensor(ls_loc, dtype=torch.float64)
        if not isinstance(ls_scale, torch.Tensor):
            ls_scale = torch.tensor(ls_scale, dtype=torch.float64)
        ls_prior = gpytorch.priors.LogNormalPrior(ls_loc + math.log(dim) / 2, ls_scale)
        ls_prior_mode = torch.exp(ls_loc - ls_scale**2)
    else:
        raise RuntimeError(
            f"Lengthscale_prior should be invgamma, gamma, or lognormal, got {lengthscale_prior}"
        )

    ls_constraint = gpytorch.constraints.GreaterThan(
        lower_bound=1e-4, transform=None, initial_value=ls_prior_mode
    )

    covar = kernel(
        lengthscale_prior=ls_prior,
        lengthscale_constraint=ls_constraint,
        ard_num_dims=dim,
        active_dims=active_dims,
    )
    if not fixed_kernel_amplitude:
        if outputscale_prior == "gamma":
            os_prior = gpytorch.priors.GammaPrior(concentration=2.0, rate=0.15)
        elif outputscale_prior == "box":
            os_prior = gpytorch.priors.SmoothedBoxPrior(a=1, b=4)
        else:
            raise RuntimeError(
                f"Outputscale_prior should be gamma or box, got {outputscale_prior}"
            )

        covar = gpytorch.kernels.ScaleKernel(
            covar,
            outputscale_prior=os_prior,
            outputscale_constraint=gpytorch.constraints.GreaterThan(1e-4),
        )
    return covar
