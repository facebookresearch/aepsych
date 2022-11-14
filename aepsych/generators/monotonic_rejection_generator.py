#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional, Sequence

import torch
from aepsych.acquisition.monotonic_rejection import MonotonicMCAcquisition
from aepsych.config import Config
from aepsych.generators.base import AEPsychGenerator
from aepsych.models.monotonic_rejection_gp import MonotonicRejectionGP
from botorch.logging import logger
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.optim.utils import columnwise_clamp, fix_features


def default_loss_constraint_fun(
    loss: torch.Tensor, candidates: torch.Tensor
) -> torch.Tensor:
    """Identity transform for constrained optimization.

    This simply returns loss as-is. Write your own versions of this
    for constrained optimization by e.g. interior point method.

    Args:
        loss (torch.Tensor): Value of loss at candidate points.
        candidates (torch.Tensor): Location of candidate points.

    Returns:
        torch.Tensor: New loss (unchanged)
    """
    return loss


class MonotonicRejectionGenerator(AEPsychGenerator[MonotonicRejectionGP]):
    """Generator specifically to be used with MonotonicRejectionGP, which generates new points to sample by minimizing
    an acquisition function through stochastic gradient descent."""

    def __init__(
        self,
        acqf: MonotonicMCAcquisition,
        acqf_kwargs: Optional[Dict[str, Any]] = None,
        model_gen_options: Optional[Dict[str, Any]] = None,
        explore_features: Optional[Sequence[int]] = None,
    ) -> None:
        """Initialize MonotonicRejectionGenerator.
        Args:
            acqf (AcquisitionFunction): Acquisition function to use.
            acqf_kwargs (Dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            model_gen_options: Dictionary with options for generating candidate, such as
                SGD parameters. See code for all options and their defaults.
            explore_features: List of features that will be selected randomly and then
                fixed for acquisition fn optimization.
        """
        if acqf_kwargs is None:
            acqf_kwargs = {}
        self.acqf = acqf
        self.acqf_kwargs = acqf_kwargs
        self.model_gen_options = model_gen_options
        self.explore_features = explore_features

    def _instantiate_acquisition_fn(self, model: MonotonicRejectionGP):
        return self.acqf(
            model=model,
            deriv_constraint_points=model._get_deriv_constraint_points(),
            **self.acqf_kwargs,
        )

    def gen(
        self,
        num_points: int,  # Current implementation only generates 1 point at a time
        model: MonotonicRejectionGP,
    ):
        """Query next point(s) to run by optimizing the acquisition function.
        Args:
            num_points (int, optional): Number of points to query.
            model (AEPsychMixin): Fitted model of the data.
        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """

        options = self.model_gen_options or {}
        num_restarts = options.get("num_restarts", 10)
        raw_samples = options.get("raw_samples", 1000)
        verbosity_freq = options.get("verbosity_freq", -1)
        lr = options.get("lr", 0.01)
        momentum = options.get("momentum", 0.9)
        nesterov = options.get("nesterov", True)
        epochs = options.get("epochs", 50)
        milestones = options.get("milestones", [25, 40])
        gamma = options.get("gamma", 0.1)
        loss_constraint_fun = options.get(
            "loss_constraint_fun", default_loss_constraint_fun
        )

        # Augment bounds with deriv indicator
        bounds = torch.cat((model.bounds_, torch.zeros(2, 1)), dim=1)
        # Fix deriv indicator to 0 during optimization
        fixed_features = {(bounds.shape[1] - 1): 0.0}
        # Fix explore features to random values
        if self.explore_features is not None:
            for idx in self.explore_features:
                val = (
                    bounds[0, idx]
                    + torch.rand(1, dtype=bounds.dtype)
                    * (bounds[1, idx] - bounds[0, idx])
                ).item()
                fixed_features[idx] = val
                bounds[0, idx] = val
                bounds[1, idx] = val

        acqf = self._instantiate_acquisition_fn(model)

        # Initialize
        batch_initial_conditions = gen_batch_initial_conditions(
            acq_function=acqf,
            bounds=bounds,
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        clamped_candidates = columnwise_clamp(
            X=batch_initial_conditions, lower=bounds[0], upper=bounds[1]
        ).requires_grad_(True)
        candidates = fix_features(clamped_candidates, fixed_features)
        optimizer = torch.optim.SGD(
            params=[clamped_candidates], lr=lr, momentum=momentum, nesterov=nesterov
        )
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )

        # Optimize
        for epoch in range(epochs):
            loss = -acqf(candidates).sum()

            # adjust loss based on constraints on candidates
            loss = loss_constraint_fun(loss, candidates)

            if verbosity_freq > 0 and epoch % verbosity_freq == 0:
                logger.info("Iter: {} - Value: {:.3f}".format(epoch, -(loss.item())))

            def closure():
                optimizer.zero_grad()
                loss.backward(
                    retain_graph=True
                )  # Variational model requires retain_graph
                return loss

            optimizer.step(closure)
            clamped_candidates.data = columnwise_clamp(
                X=clamped_candidates, lower=bounds[0], upper=bounds[1]
            )
            candidates = fix_features(clamped_candidates, fixed_features)
            lr_scheduler.step()

        # Extract best point
        with torch.no_grad():
            batch_acquisition = acqf(candidates)
        best = torch.argmax(batch_acquisition.view(-1), dim=0)
        Xopt = candidates[best][:, :-1].detach()
        return Xopt

    @classmethod
    def from_config(cls, config: Config):
        classname = cls.__name__
        acqf = config.getobj("common", "acqf", fallback=None)
        extra_acqf_args = cls._get_acqf_options(acqf, config)

        options = {}
        options["num_restarts"] = config.getint(classname, "restarts", fallback=10)
        options["raw_samples"] = config.getint(classname, "samps", fallback=1000)
        options["verbosity_freq"] = config.getint(
            classname, "verbosity_freq", fallback=-1
        )
        options["lr"] = config.getfloat(classname, "lr", fallback=0.01)  # type: ignore
        options["momentum"] = config.getfloat(classname, "momentum", fallback=0.9)  # type: ignore
        options["nesterov"] = config.getboolean(classname, "nesterov", fallback=True)
        options["epochs"] = config.getint(classname, "epochs", fallback=50)
        options["milestones"] = config.getlist(
            classname, "milestones", fallback=[25, 40]  # type: ignore
        )
        options["gamma"] = config.getfloat(classname, "gamma", fallback=0.1)  # type: ignore
        options["loss_constraint_fun"] = config.getobj(
            classname, "loss_constraint_fun", fallback=default_loss_constraint_fun
        )

        explore_features = config.getlist(classname, "explore_idxs", fallback=None)  # type: ignore

        return cls(
            acqf=acqf,
            acqf_kwargs=extra_acqf_args,
            model_gen_options=options,
            explore_features=explore_features,
        )
