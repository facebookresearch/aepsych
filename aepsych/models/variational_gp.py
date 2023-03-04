#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from aepsych.config import Config
from aepsych.likelihoods.ordinal import OrdinalLikelihood
from aepsych.models.base import AEPsychModel
from aepsych.models.utils import get_probability_space, select_inducing_points
from aepsych.utils import promote_0d
from botorch.models import SingleTaskVariationalGP
from gpytorch.likelihoods import BernoulliLikelihood, BetaLikelihood
from gpytorch.mlls import VariationalELBO


# TODO: Find a better way to do this on the Ax/Botorch side
class MyHackyVariationalELBO(VariationalELBO):
    def __init__(self, likelihood, model, beta=1.0, combine_terms=True):
        num_data = model.model.train_targets.shape[0]
        super().__init__(likelihood, model.model, num_data, beta, combine_terms)


class VariationalGP(AEPsychModel, SingleTaskVariationalGP):
    @classmethod
    def get_mll_class(cls):
        return MyHackyVariationalELBO

    @classmethod
    def construct_inputs(cls, training_data, **kwargs):
        inputs = super().construct_inputs(training_data=training_data, **kwargs)

        inducing_size = kwargs.get("inducing_size")
        inducing_point_method = kwargs.get("inducing_point_method")
        bounds = kwargs.get("bounds")
        inducing_points = select_inducing_points(
            inducing_size,
            inputs["covar_module"],
            inputs["train_X"],
            bounds,
            inducing_point_method,
        )

        inputs.update(
            {
                "inducing_points": inducing_points,
            }
        )

        return inputs

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None) -> Dict:
        classname = cls.__name__

        options = super().get_config_options(config, classname)

        inducing_point_method = config.get(
            classname, "inducing_point_method", fallback="auto"
        )
        inducing_size = config.getint(classname, "inducing_size", fallback=10)
        learn_inducing_points = config.getboolean(
            classname, "learn_inducing_points", fallback=False
        )

        options.update(
            {
                "inducing_size": inducing_size,
                "inducing_point_method": inducing_point_method,
                "learn_inducing_points": learn_inducing_points,
            }
        )

        return options


class BinaryClassificationGP(VariationalGP):
    stimuli_per_trial = 1
    outcome_type = "binary"

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool, optional): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            post = self.posterior(x)

        if probability_space:
            fmean, fvar = get_probability_space(
                likelihood=self.likelihood, posterior=post
            )
        else:
            fmean = post.mean.squeeze()
        fvar = post.variance.squeeze()

        return promote_0d(fmean), promote_0d(fvar)

    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None):
        options = super().get_config_options(config)
        if options["likelihood"] is None:
            options["likelihood"] = BernoulliLikelihood()
        return options
    
class BetaRegressionGP(VariationalGP):
    outcome_type = "percentage"

    def predict(
        self, x: Union[torch.Tensor, np.ndarray], probability_space: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance.

        Args:
            x (torch.Tensor): Points at which to predict from the model.
            probability_space (bool, optional): Return outputs in units of
                response probability instead of latent function value. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        with torch.no_grad():
            post = self.posterior(x)

        if probability_space:
            fmean, fvar = get_probability_space(
                likelihood=self.likelihood, posterior=post
            )
        else:
            fmean = post.mean.squeeze()
        fvar = post.variance.squeeze()

        return promote_0d(fmean), promote_0d(fvar)
    
    @classmethod
    def get_config_options(cls, config: Config, name: Optional[str] = None):
        options = super().get_config_options(config)
        if options["likelihood"] is None:
            options["likelihood"] = BetaLikelihood()
            ordinal_likelihood = OrdinalLikelihood()
        return options