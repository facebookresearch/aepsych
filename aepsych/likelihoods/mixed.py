#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from aepsych.config import ConfigurableMixin
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import Likelihood
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from linear_operator.operators import DiagLinearOperator


class ListLikelihood(Likelihood, ConfigurableMixin):
    def __init__(self, likelihoods: list[Likelihood]) -> None:
        """A wrapper around a list of likelihoods. Currently, the first
        likelihood is given special treatment and any methods/functions that
        would normally be called on the likelihood are called on the first. The
        only special thing this class does is allow a list of likelihood to be
        iterated over and indexed.

        Args:
            likelihoods (list[Likelihood]): A list of likelihoods where the
                first is the primary one.

        """
        super().__init__()
        self.likelihoods = torch.nn.ModuleList(likelihoods)

    # def __getattr__(self, name):
    #     if name == "likelihoods":
    #         return self.likelihoods
    #     else:
    #         return getattr(self.likelihoods[0], name)

    def __iter__(self):
        return iter(self.likelihoods)

    def __getitem__(self, key):
        return self.likelihoods[key]

    def forward(self, *args, **kwargs):
        """Forward pass for the likelihood. This is a no-op for this class.
        When a forward is called on a class instance, the forward of the first
        likelihood is called instead."""
        return self.likelihoods[0].forward(*args, **kwargs)


class MixedVariationalELBO(_ApproximateMarginalLogLikelihood):
    def __init__(
        self,
        likelihood: ListLikelihood,
        task_indices: list[torch.Tensor],
        model: ApproximateGP,
        num_data: int,
        beta: float = 1.0,
        combine_terms: bool = True,
    ) -> None:
        """A variant of the variational ELBO that takes multiple likelihoods.
        Only the first likelihood is learnable. Used for adding constraints to
        variational models.

        Args:
        likelihood (ListLikelihood): An instance of the ListLikelihood class
            that represents a list of likelihoods. Only the first likelihood is
            learnable.
            All likelihoods have to be univariate one-dimensional likelihood.
        task_indices (list[torch.Tensor]): A list of tensors of indices that
            correspond to each task/data type being modeled, one for each
            likelihood.
        model (gpytorch.models.ApproximateGP): The approximate GP model
        num_data (int): The total number of training data points (necessary for
            SGD)
        beta (float): A multiplicative factor for the KL divergence term.
            Setting it to 1 (default) recovers true variational inference.
            Setting it to anything less than 1 reduces the regularization effect
            of the model.
        combine_terms (bool): Whether or not to sum the expected NLL with the
            KL terms (default True)
        """
        super().__init__(
            likelihood=likelihood[0],
            model=model,
            num_data=num_data,
            beta=beta,
            combine_terms=combine_terms,
        )

        self.list_likelihoods = likelihood
        self.task_indices = task_indices

    def _log_likelihood_term(
        self, variational_dist_f: MultivariateNormal, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        mean = variational_dist_f.loc
        variance = variational_dist_f.variance

        """
        Linear Operator is not very efficient in indexing.
        For instance, covar[indices.unsqueeze(-1), indices.unsqueeze(-2)]
        consumes an unnecessary amount of memory.
        Therefore, we opt in for the following hacks.
        """
        res = torch.tensor(0.0).to(mean)
        for likelihood, indices in zip(self.list_likelihoods, self.task_indices):
            res = res + likelihood.expected_log_prob(
                target[indices],
                MultivariateNormal(
                    mean[indices],
                    DiagLinearOperator(diag=variance[indices]),
                ),
                **kwargs,
            ).sum(-1)

        return res

    def forward(
        self, variational_dist_f: MultivariateNormal, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """
        Computes the Variational ELBO given the variational distribution and
        the target data. Calling this function will call this class's
        log_likelihood_term, which will sum over the expected log probabilities
        of each likelihood and its corresponding task.

        Args:
        variational_dist_f (gpytorch.distributions.MultivariateNormal): The
            variational distribution of the latent function.
        target (Tensor): The target data.
        **kwargs: Additional arguments passed to the likelihood's
            expected_log_prob. These kwargs are passed to every likelihood in
            ListLikelihood.

        Returns:
            torch.Tensor: The variational ELBO. Output shape corresponds to
                batch shape of the model/input data.
        """
        return super().forward(variational_dist_f, target, **kwargs)
