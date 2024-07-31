#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gpytorch

import torch
from gpytorch.likelihoods import _OneDimensionalLikelihood

from aepsych.distributions import RTDistWithUniformLapseRate

class DDMLikelihood(_OneDimensionalLikelihood):
    """ """

    def __init__(self, distribution, max_shift = None, restrict_skew = False):
        super().__init__()
        self.distribution = distribution
        self.register_parameter(
            name="raw_relative_x0", parameter=torch.nn.Parameter(torch.randn(1))
        )
        self.register_constraint("raw_relative_x0", gpytorch.constraints.Interval(0, 1))

        self.register_parameter(
            name="raw_t0", parameter=torch.nn.Parameter(torch.randn(1))
        )
        self.register_constraint("raw_t0", gpytorch.constraints.Interval(0., 1.0))

        self.register_parameter(
            name="raw_threshold", parameter=torch.nn.Parameter(torch.randn(1))
        )
        self.register_constraint("raw_threshold", gpytorch.constraints.Positive())

        self.max_shift = max_shift
        self.restrict_skew = restrict_skew

    def _set_relative_x0(self, value):
        value = self.raw_relative_x0_constraint.inverse_transform(value)
        self.initialize(raw_relative_x0=value)

    def _set_threshold(self, value):
        value = self.raw_threshold_constraint.inverse_transform(value)
        self.initialize(raw_threshold=value)

    def _set_t0(self, value):
        value = self.raw_t0_constraint.inverse_transform(value)
        self.initialize(raw_t0=value)

    @property
    def relative_x0(self):
        return self.raw_relative_x0_constraint.transform(self.raw_relative_x0)

    @relative_x0.setter
    def relative_x0(self, value):
        self._set_relative_x0(value)

    @property
    def x0(self):
        return self.threshold * (2*self.relative_x0 - 1)

    @property
    def t0(self):
        return self.raw_t0_constraint.transform(self.raw_t0)

    @t0.setter
    def t0(self, value):
        self._set_t0(value)

    @property
    def threshold(self):
        return self.raw_threshold_constraint.transform(self.raw_threshold)

    @threshold.setter
    def threshold(self, value):
        self._set_threshold(value)

    def forward(self, function_samples, *params, **kwargs):
        return self.distribution(
            drift=function_samples, threshold=self.threshold, relative_x0=self.relative_x0, t0=self.t0, max_shift = self.max_shift, restrict_skew = self.restrict_skew
        )

    @classmethod
    def from_config(cls, config):
        classname = cls.__name__
        max_shift = config.getfloat(classname, "max_shift", fallback=None)
        restrict_skew = config.getboolean(classname, "restrict_skew", fallback=None)

        distribution = config.getobj(classname, "distribution")


        return cls(distribution=distribution, max_shift=max_shift, restrict_skew=restrict_skew)

    # def log_marginal(self, observations, function_dist, *args, **kwargs):
    #     """
    #     here we need the expectation of logp(r,c|f) w.r.t f
    #     p(r, c|f) = p(r|c,f)p(c|f), so we can factorize
    #     the log marginal as E_f log p(r|c,f) + E_f log p(c|f).
    #     and to the integrals separately
    #     """
    #     choices = observations > 0
    #     # rt_log_probs = torch.where(choices, yes_log_probs, no_log_probs)

    #     def choice_prob_sampler(function_samples):
    #         ddmdist = self.forward(function_samples)
    #         return ddmdist.choice_dist.log_prob(choices.float()).exp()

    #     choice_marginal = self.quadrature(choice_prob_sampler, function_dist)

    #     def rt_prob_sampler(function_samples):
    #         ddmdist = self.forward(function_samples)
    #         yes_probs = ddmdist.rt_yes_dist.log_prob(torch.abs(observations)).exp()
    #         no_probs = ddmdist.rt_no_dist.log_prob(torch.abs(observations)).exp()
    #         return torch.where(choices, yes_probs, no_probs)

    #     rt_marginal = self.quadrature(rt_prob_sampler, function_dist)

    #     return choice_marginal.log() + rt_marginal.log()


class LapseRateRTLikelihood(_OneDimensionalLikelihood):
    def __init__(self, base_likelihood, max_rt=10.0):
        super().__init__()
        self.max_rt = max_rt
        self.base_likelihood = base_likelihood
        self.register_parameter(
            name="raw_lapse_rate", parameter=torch.nn.Parameter(torch.randn(1))
        )
        self.register_constraint(
            "raw_lapse_rate", gpytorch.constraints.Interval(1e-5, 0.2)
        )  # any greater than that and the model is really bad anyway

    @property
    def lapse_rate(self):
        return self.raw_lapse_rate_constraint.transform(self.raw_lapse_rate)

    def forward(self, function_samples, *args, **kwargs):
        base_dist = self.base_likelihood(function_samples, *args, **kwargs)
        return RTDistWithUniformLapseRate(
            lapse_rate=self.lapse_rate, base_dist=base_dist, max_rt=self.max_rt
        )

    @classmethod
    def from_config(cls, config):
        classname = cls.__name__
        max_rt = config.getfloat(classname, "max_rt", fallback=10.)

        base_lik_class = config.getobj(classname, "base_likelihood")

        base_lik = base_lik_class.from_config(config)
        return cls(base_likelihood = base_lik, max_rt = max_rt)
