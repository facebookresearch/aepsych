from __future__ import annotations

import warnings

from copy import deepcopy
from itertools import product

from typing import Iterable, List, Optional, Tuple, Union

import botorch

import gpytorch

import numpy as np
import torch
from aepsych.models import GPClassificationModel
from aepsych.models.utils import select_inducing_points

from gpytorch.variational import CholeskyVariationalDistribution


class MixedInputGPClassificationModel(GPClassificationModel):
    """Mixed input classification GP. This model can fit both continuous inputs (e.g. contrast,
    frequency, etc) and discrete inputs (e.g. subject ID). The covariance between
    two points is defined as k_x(x, x') + k_t[i, j] + k_x(x, x') * k_t[i, j] where k(x, x')
    is the usual GP kernel and k_t[i, j] is a product of independent estimated covariances
    between the values of each discrete input and can be of potentially low rank.

    With discrete_kernel="categorical", this model is the classification analogue of
    `botorch.models.gp_regression_mixed.MixedSingleTaskGP`. With discrete_kernel="index",
    it is the classification analogue of Hadamard Multitask GP regression in
    https://docs.gpytorch.ai/en/stable/examples/03_Multitask_Exact_GPs/Hadamard_Multitask_GP_Regression.html
    """

    _num_outputs = 1
    _batch_shape = torch.Size([1])
    stimuli_per_trial = 1
    outcome_type = "binary"

    def __init__(
        self,
        discrete_param_levels: List[int],
        discrete_param_ranks: List[int],
        discrete_induc_method="all",
        discrete_kernel="index",
        *args,
        **kwargs,
    ):
        """Initialize mixed-input GP Classification model

        Args:
            discrete_param_levels (List[int]): List containing the number of levels for each discrete input.
            discrete_param_ranks (List[int]): List containing the rank of the covariance to estimate for each discrete input.
            discrete_induc_method (str, optional): Method for selecting inducing points for the discrete inputs.
                Options are "data", "random_subset" and "all" (see notes). Defaults to "all".
            discrete_kernel (str, optional): Representation for the discrete input covariance. Options are "index" and "categorical"
                (see notes). Defaults to "index".
            Additional arguments are passed to `GPClassificationModel`.

        Notes:
            `discrete_induc_method` can be "all", "random_subset", or "data". "data" means place inducing points at the observations
            (both continuous and discrete). "random_subset" means we select a random permutation of discrete inputs for each
            continuous inducing point. "all" means that we replicate the continuous inducing points for each permutation of the
            discrete inducing points. This latter method is likely most accurate but scales poorly with the number of discrete
            inputs and their values, and is only recommended for offline analysis. For both "all" and "random_subset", continuous
            inducing points are selected according to the `inducing_point_method` passed to the GPClassification superclass.

            `discrete_kernel` can be "index", which uses `gpytorch.kernels.IndexKernel` (a freeform, potentially
            low-rank covariance) or "categorical", which uses `botorch.models.kernels.CategoricalKernel` (a kernel
            based on Hamming distances, i.e. the number of discrete features on which inputs match).
        """

        super().__init__(*args, **kwargs)

        assert len(discrete_param_levels) == len(
            discrete_param_ranks
        ), "discrete parameter levels and ranks should match in length"

        assert discrete_kernel in (
            "index",
            "categorical",
        ), "only index or categorical kernels supported for discrete kernel"

        self.discrete_induc_method = discrete_induc_method
        if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel):
            warnings.warn(
                "Continuous kernel should not include a scale, since discrete kernel includes it"
                + "removing scale from kernel"
            )
            continuous_covar = self.covar_module.base_kernel
        else:
            continuous_covar = self.covar_module

        # continuous covar
        continuous_covar.active_dims = torch.arange(self.dim)
        continuous_covar.ard_num_dims = self.dim

        # discrete covar
        self.n_discrete = len(discrete_param_levels)
        self.discrete_param_levels = torch.Tensor(discrete_param_levels)

        discrete_covars = []
        discrete_input_idx = 0
        for n_discrete, ranks in zip(discrete_param_levels, discrete_param_ranks):
            if discrete_kernel == "categorical":
                newkern = botorch.models.kernels.CategoricalKernel(
                    active_dims=(self.dim + discrete_input_idx,),
                    ard_num_dims=1,
                )
            elif discrete_kernel == "index":
                newkern = gpytorch.kernels.IndexKernel(
                    num_tasks=n_discrete,
                    rank=ranks,
                    active_dims=(self.dim + discrete_input_idx,),
                    ard_num_dims=1,
                    prior=gpytorch.priors.LKJCovariancePrior(
                        n=n_discrete,
                        eta=1.5,
                        sd_prior=gpytorch.priors.GammaPrior(1.0, 0.15),
                    ),
                )
            discrete_covars.append(newkern)
            discrete_input_idx += 1

        self.discrete_covars = discrete_covars
        self.continuous_covar = continuous_covar
        discrete_covars_copy = deepcopy(discrete_covars)
        self.covar_module = gpytorch.kernels.AdditiveKernel(
            continuous_covar, *discrete_covars
        ) + gpytorch.kernels.ProductKernel(continuous_covar, *discrete_covars_copy)

    def forward(self, X: torch.Tensor):

        n_continuous = X.shape[-1] - self.n_discrete

        X_continuous, X_discrete = torch.split(
            X, [n_continuous, self.n_discrete], dim=-1
        )
        transformed_X = self.normalize_inputs(X_continuous)
        transformed_inputs = torch.cat((transformed_X, X_discrete), dim=-1)
        mean_x = self.mean_module(transformed_inputs)
        covar = self.covar_module(transformed_inputs)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

    @property
    def train_inputs(self):
        return (torch.cat((self.X_continuous, self.X_discrete), -1),)

    # TODO don't violate LSP by having a proper base class
    def set_train_data(  # type: ignore
        self,
        X_continuous: torch.Tensor,
        X_discrete: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.X_continuous = X_continuous
        self.train_targets = targets
        self.X_discrete = X_discrete

    # TODO don't violate LSP by having a proper base class
    def fit(  # type: ignore
        self,
        X_continuous: torch.Tensor,
        X_discrete: torch.Tensor,
        y: torch.Tensor,
        **kwargs,
    ):
        self.train()

        X_continuous, X_discrete = self._promote_input_dims(X_continuous, X_discrete)

        self.set_train_data(X_continuous, X_discrete, y)

        discrete_param_indices = [np.arange(i) for i in self.discrete_param_levels]

        # place inducing points for VI
        if self.discrete_induc_method == "data":
            # just induce on the data
            inducing_points = torch.cat((X_continuous, X_discrete), -1)
        elif self.discrete_induc_method == "random_subset":
            # place continuous inducing points on a random subset
            # of the discrete indices
            all_discrete_params = np.r_[list(product(*discrete_param_indices))]
            indices = np.arange(all_discrete_params.shape[0])
            chosen_indices = np.random.choice(indices, self.inducing_size)
            discrete_induc = all_discrete_params[chosen_indices]
            continuous_induc = select_inducing_points(
                inducing_size=self.inducing_size,
                covar_module=self.continuous_covar,
                X=X_continuous,
                method=self.inducing_point_method,
            )
            inducing_points = torch.Tensor(np.c_[continuous_induc, discrete_induc])
        elif self.discrete_induc_method == "all":
            # place continuous inducing points for each permutation
            # of the discrete indices
            induc = []
            for discrete_config in product(*discrete_param_indices):
                local_X = X_continuous[
                    (X_discrete == torch.Tensor(discrete_config)).all(-1)
                ]
                local_induc = select_inducing_points(
                    inducing_size=self.inducing_size,
                    covar_module=self.continuous_covar,
                    X=local_X,
                    method=self.inducing_point_method,
                )
                n_local_induc = local_induc.shape[0]
                disc_induc = torch.tile(
                    torch.Tensor(discrete_config), (n_local_induc, 1)
                )
                induc.append(torch.hstack([local_induc, disc_induc]))

            inducing_points = torch.cat(induc, dim=0)
        else:
            raise RuntimeError(
                f"Unknown inducing point method {self.discrete_induc_method}!"
            )

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.shape[0]
        )
        self.variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=False,
        )

        n = X_continuous.shape[0]
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, n)

        self._fit_mll(mll, **kwargs)

    def _promote_input_dims(self, X_continuous: torch.Tensor, X_discrete: torch.Tensor):
        # make sure inputs are always matrix-valued
        if len(X_discrete.shape) == 1:
            X_discrete = X_discrete[:, None]
        if len(X_continuous.shape) == 1:
            X_continuous = X_continuous[:, None]
        return X_continuous, X_discrete

    # TODO don't violate LSP by having a proper base class
    def predict(  # type: ignore
        self,
        x: Union[torch.Tensor, np.ndarray],
        discrete_config: Union[torch.Tensor, np.ndarray],
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict from mixed-input model. Note that this prediction method generates
        continuous predictions at a single discrete parameter configuration (e.g.
        for one participant) -- you should call this in a loop to generate predictions
        for all discrete values.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Continuous input.
            discrete_config (Union[torch.Tensor, np.ndarray]): A single discfete configuration
                of parameters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean, variance of the GP at the target locations.
        """
        discrete_config = torch.as_tensor(discrete_config)
        x = torch.as_tensor(x)
        self._promote_input_dims(x, discrete_config)

        xgridsize = x.shape[0]
        discrete_grid = torch.tile(discrete_config, (xgridsize, 1))
        x_aug = torch.hstack([x, discrete_grid])

        return super().predict(x_aug, *args, **kwargs)
