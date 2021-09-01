#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union

import numpy as np
import torch
from aepsych.acquisition.monotonic_rejection import (
    MonotonicMCAcquisition,
    MonotonicMCLSE,
)
from aepsych.config import Config
from aepsych.modelbridge.base import ModelBridge, _prune_extra_acqf_args
from aepsych.models.monotonic_rejection_gp import MonotonicGPLSETS, MonotonicRejectionGP

ModelType = TypeVar("ModelType", bound="MonotonicSingleProbitModelbridge")


class MonotonicSingleProbitModelbridge(ModelBridge):
    """
    Modelbridge wrapping monotonic single probit GP models.

    Attributes:
        outcome_type: fixed to "single_probit".

    """

    outcome_type = "single_probit"

    def __init__(
        self,
        lb: Union[np.ndarray, torch.Tensor],
        ub: Union[np.ndarray, torch.Tensor],
        samps: int = 1000,
        dim: int = 1,
        acqf: Optional[MonotonicMCAcquisition] = None,
        extra_acqf_args: Optional[Dict[str, object]] = None,
        monotonic_idxs: Optional[Sequence[int]] = None,
        uniform_idxs: Optional[Sequence[int]] = None,
        model: Optional[MonotonicRejectionGP] = None,
    ) -> None:
        """Initialize monotonic modelbridge.

        Args:
            lb (Union[np.ndarray, torch.Tensor]): Lower bounds of domain.
            ub (Union[np.ndarray, torch.Tensor]): Upper bounds of domain.
            samps (int, optional): Number of quasi-random samples to use for finding
                an initial condition for acquisition function optimization. Defaults to 1000.
            dim (int, optional): Number of input dimensions. Defaults to 1.
            acqf (MonotonicMCAcquisition, optional): Acquisition function
                to use. Defaults to whatver the underlying model defaults to.
            extra_acqf_args (Dict[str, object], optional): Extra arguments to
                pass to acquisition function. Defaults to no arguments.
            monotonic_idxs (Sequence[int], optional): Dimensions to constrain to be
                monotonically increasing. Defaults no dimensions.
            uniform_idxs (Sequence[int], optional): Dimensions to sample uniformly
                instead of using acquisition function (this can sometimes help improve
                exploration). Defaults to no dimensions.
            model (MonotonicRejectionGP, optional): Underyling model to use.
                Defaults to MonotonicGPLSETS.
        """
        super().__init__(
            lb=lb, ub=ub, dim=dim, acqf=acqf, extra_acqf_args=extra_acqf_args
        )

        if extra_acqf_args is None:
            extra_acqf_args = {}
        if monotonic_idxs is None:
            monotonic_idxs = []
        if uniform_idxs is None:
            uniform_idxs = []
        self.target = extra_acqf_args.get("target", None)
        self.samps = samps
        self.uniform_idxs = uniform_idxs

        self.model = model or MonotonicGPLSETS(
            likelihood="probit-bernoulli",
            monotonic_idxs=monotonic_idxs,
            target_value=self.target,  # type: ignore
        )
        # self.acqf was set by super() so we just copy it to model
        assert issubclass(self.acqf, MonotonicMCAcquisition), (
            "Monotonic models require subclass of MonotonicMCAcquisition"
            + f"(got {self.acqf})"
        )
        self.model.acqf = self.acqf

    def _get_acquisition_fn(self) -> MonotonicMCAcquisition:
        return self.model._get_acquisition_fn()

    def gen(self, num_points: int = 1, use_uniform: bool = False) -> np.ndarray:
        """Query next point(s) to run by optimizing the acquisition function.

        Args:
            num_points (int, optional): Number of points to query. Defaults to 1.
            use_uniform (bool, optional): If true, sample self.uniform_idxs uniformly
                instead of acquiring over them. Defaults to False.

        Raises:
            NotImplementedError: If requesting num_points > 1 for a model other
                than GPLSETS.

        Returns:
            np.ndarray: Next set of point(s) to evaluate, [num_points x dim].
        """
        explore_strat_features: Optional[Sequence[int]]
        gen_args = {"raw_samples": self.samps}
        if use_uniform and len(self.uniform_idxs) > 0:
            explore_strat_features = self.uniform_idxs
        else:
            explore_strat_features = None

        if num_points > 1:
            if isinstance(self.model, MonotonicGPLSETS):
                next_pts, _ = self.model.gen(
                    n=num_points,
                    model_gen_options=gen_args,
                    explore_features=explore_strat_features,
                )
            else:
                raise NotImplementedError(
                    "batch gen only supported for MonotonicGPLSETS!"
                )
        else:
            next_pts, _ = self.model.gen(
                model_gen_options=gen_args, explore_features=explore_strat_features,
            )

        return next_pts.numpy()

    def predict(
        self, x: Union[np.ndarray, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Call underlying model for prediction.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Points at which to predict.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at x.
        """
        # MGPLSETS expects batch x dim so augment if needed
        if len(x.shape) == 1:
            x = x[:, None]
        return self.model.predict(torch.Tensor(x))

    def sample(self, x: torch.Tensor, num_samples: int, **kwargs) -> torch.Tensor:
        """Sample from underlying model.

        Args:
            x (torch.Tensor): Points at which to sample.
            num_samples (int): How many points to draw.
            Additional arguments are passed to underlying model.

        Returns:
            torch.Tensor: Posterior samples [num_samples x dim]
        """
        return self.model.sample(x, **kwargs)

    def fit(self, train_x: torch.Tensor, train_y: torch.LongTensor):
        """Fit underlying model.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
        """
        self.model.fit(train_x, train_y, np.c_[self.lb, self.ub])

    def update(self, train_x: torch.Tensor, train_y: torch.LongTensor):
        """Update underlying model by warm starting from previous variational fit.

        Args:
            train_x (torch.Tensor): Inputs.
            train_y (torch.LongTensor): Responses.
        """

        # update the warm-start model fitting
        self.model.update(train_x, train_y)

    @classmethod
    def from_config(cls: Type[ModelType], config: Config) -> ModelType:
        """Alternate constructor for monotonic modelbridge from AEPsych config.

        Args:
            config (Config): Config containing params for this object and child objects.

        Returns:
            MonotonicSingleProbitModelbridge: Configured class instance.
        """

        classname = cls.__name__
        model_cls = config.getobj("experiment", "model", fallback=MonotonicRejectionGP)
        mean_covar_factory = config.getobj(model_cls.__name__, "mean_covar_factory")
        monotonic_idxs: List[int] = config.getlist(
            model_cls.__name__, "monotonic_idxs", fallback=[-1]
        )
        uniform_idxs: List[int] = config.getlist(
            model_cls.__name__, "uniform_idxs", fallback=[]
        )

        mean, covar = mean_covar_factory(config)
        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        samps = config.getint(classname, "samps", fallback=1000)
        assert lb.shape[0] == ub.shape[0], "bounds are of different shapes!"
        dim = lb.shape[0]
        for idx in monotonic_idxs:
            assert (
                idx < dim
            ), f"monotonic_idx {int(idx)} is out of bounds for dimensionality {dim}"
        acqf = config.getobj("experiment", "acqf", fallback=MonotonicMCLSE)
        acqf_name = acqf.__name__

        # note: objective isn't extra_args for monotonicgp
        default_extra_acqf_args = {"beta": 3.98, "target": 0.75}
        extra_acqf_args = {
            k: config.getobj(acqf_name, k, fallback_type=float, fallback=v, warn=False)
            for k, v in default_extra_acqf_args.items()
        }
        extra_acqf_args = _prune_extra_acqf_args(acqf, extra_acqf_args)
        if (
            "objective" in extra_acqf_args.keys()
            and extra_acqf_args["objective"] is not None
        ):
            extra_acqf_args["objective"] = extra_acqf_args["objective"]()

        if issubclass(model_cls, MonotonicGPLSETS):
            model = model_cls(
                likelihood="probit-bernoulli",
                monotonic_idxs=monotonic_idxs,
                target_value=extra_acqf_args.get("target", 0.75),
                extra_acqf_args=extra_acqf_args,
                mean_module=mean,
                covar_module=covar,
            )
        else:
            model = model_cls(
                likelihood="probit-bernoulli",
                monotonic_idxs=monotonic_idxs,
                extra_acqf_args=extra_acqf_args,
                mean_module=mean,
                covar_module=covar,
            )

        return cls(
            lb=lb,
            ub=ub,
            samps=samps,
            dim=dim,
            acqf=acqf,
            extra_acqf_args=extra_acqf_args,
            monotonic_idxs=monotonic_idxs,
            uniform_idxs=uniform_idxs,
            model=model,
        )
