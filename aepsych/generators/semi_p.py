#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, Type

import torch
from aepsych.acquisition.objective.semi_p import SemiPThresholdObjective
from aepsych.generators import OptimizeAcqfGenerator
from aepsych.models.semi_p import SemiParametricGPModel


class IntensityAwareSemiPGenerator(OptimizeAcqfGenerator):
    """Generator for SemiP. With botorch machinery, in order to optimize acquisition
    separately over context and intensity, we need two ingredients.
    1. An objective that samples from some posterior w.r.t. the context. From the
        paper, this is ThresholdBALV and needs the threshold posterior.
        `SemiPThresholdObjective` implements this for ThresholdBALV but theoretically
        this can be any subclass of `SemiPObjectiveBase`.
    2. A way to do acquisition over context and intensity separately, which is
        provided by this class. We optimize the acquisition function over context
        dimensions, then conditioned on the optimum we evaluate the intensity
        at the objective to obtain the intensity value.

    We only developed ThresholdBALV that is specific to SemiP, which is what we tested
    with this generator. It should work with other similar acquisition functions.
    """

    def gen(  # type: ignore[override]
        self,
        num_points: int,
        model: SemiParametricGPModel,  # type: ignore[override]
        context_objective: Type = SemiPThresholdObjective,
        fixed_features: Optional[Dict[int, float]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Query next point(s) to run by optimizing the acquisition function for both context and intensity.

        Args:
            num_points (int): Number of points to query.
            model (SemiParametricGPModel): Fitted semi-parametric model of the data.
            context_objective (Type): The objective function used for context. Defaults to SemiPThresholdObjective.
            fixed_features (Dict[int, float], optional): Not implemented for this generator.
            **kwargs: Passed to generator

        Returns:
            torch.Tensor: Next set of point(s) to evaluate, [num_points x dim].
        """
        if fixed_features is not None:
            raise ValueError("Fixed features not supported for semi_p generators")

        fixed_features_ = {model.stim_dim: 0.0}
        next_x = super().gen(
            num_points=num_points, model=model, fixed_features=fixed_features_, **kwargs
        )
        # to compute intensity, we need the point where f is at the
        # threshold as a function of context. self.acqf_kwargs should contain
        # remaining objective args (like threshold target value)

        thresh_objective = context_objective(
            likelihood=model.likelihood, stim_dim=model.stim_dim, **self.acqf_kwargs
        )
        kc_mean_at_best_context = model(torch.Tensor(next_x)).mean
        thresh_at_best_context = thresh_objective(kc_mean_at_best_context)
        thresh_at_best_context = torch.clamp(
            thresh_at_best_context,
            min=model.lb[model.stim_dim],
            max=model.ub[model.stim_dim],
        )
        next_x[..., model.stim_dim] = thresh_at_best_context.detach()
        return next_x
