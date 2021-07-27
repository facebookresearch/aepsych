#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from aepsych.modelbridge.base import ModelBridge, _prune_extra_acqf_args
from aepsych.acquisition.monotonic_rejection import (
    MonotonicMCAcquisition,
    MonotonicMCLSE,
)
from aepsych.models.monotonic_rejection_gp import MonotonicGPLSETS, MonotonicRejectionGP


class MonotonicSingleProbitModelbridge(ModelBridge):
    """
    Minimal shim for MonotonicGPLSE

    This is basically hacky multiple inheritance, it is definitely bad.
    TODO before opensourcing we need to standardize interfaces so that this shim goes away

    """

    outcome_type = "single_probit"

    def __init__(
        self,
        lb,
        ub,
        samps=1000,
        dim=1,
        acqf=None,
        extra_acqf_args=None,
        monotonic_idxs=None,
        uniform_idxs=None,
        model=None,
    ):
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
            target_value=self.target,
        )
        # self.acqf was set by super() so we just copy it to model
        assert issubclass(self.acqf, MonotonicMCAcquisition), (
            "Monotonic models require subclass of MonotonicMCAcquisition"
            + f"(got {self.acqf})"
        )
        self.model.acqf = self.acqf

    def _get_acquisition_fn(self) -> MonotonicMCAcquisition:
        return self.model._get_acquisition_fn()

    def gen(self, num_points=1, use_uniform=False, **kwargs):
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
                    **kwargs,
                )
            else:
                raise NotImplementedError(
                    "batch gen only supported for MonotonicGPLSETS!"
                )
        else:
            next_pts, _ = self.model.gen(
                model_gen_options=gen_args,
                explore_features=explore_strat_features,
                **kwargs,
            )

        return next_pts.numpy()

    def predict(self, x):
        # MGPLSETS expects batch x dim so augment if needed
        if len(x.shape) == 1:
            x = x[:, None]
        return self.model.predict(x)

    def sample(self, x, **kwargs):
        return self.model.sample(x, **kwargs)

    def fit(self, train_x, train_y):
        self.model.fit(train_x, train_y, np.c_[self.lb, self.ub])

    def update(self, train_x, train_y):
        # update the warm-start model fitting
        self.model.update(train_x, train_y)

    @classmethod
    def from_config(cls, config):

        classname = cls.__name__
        model_cls = config.getobj("experiment", "model", fallback=MonotonicRejectionGP)
        mean_covar_factory = config.getobj(model_cls.__name__, "mean_covar_factory")
        monotonic_idxs = config.getlist(
            model_cls.__name__, "monotonic_idxs", fallback=[-1]
        )
        uniform_idxs = config.getlist(model_cls.__name__, "uniform_idxs", fallback=[])

        mean, covar = mean_covar_factory(config)
        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        samps = config.getint(classname, "samps", fallback=1000)
        assert lb.shape[0] == ub.shape[0], "bounds are of different shapes!"
        dim = lb.shape[0]

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
