#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import warnings


from logging import getLogger
from numbers import Number

import torch
from torch.distributions import Bernoulli, Exponential, LogNormal, Normal
from gpytorch import constraints
from gpytorch.distributions import Distribution
from torch.distributions import (
    Gamma,
    TransformedDistribution,
    Uniform,
)
from torch.distributions.transforms import AffineTransform, ExpTransform, PowerTransform
from torch.distributions.utils import broadcast_all

logger = getLogger()


class RTDistWithUniformLapseRate(Distribution):
    arg_constraints = {
        "lapse_rate": constraints.Interval(1e-5, 0.2),
        "max_rt": constraints.Positive(),
    }

    def __init__(
        self, lapse_rate, max_rt, base_dist, validate_args=False, **kwargs
    ):
        self.lapse_rate = lapse_rate
        self.base_dist = base_dist
        self.max_rt = max_rt
        self.lapse_dist = Uniform(-self.max_rt, self.max_rt)
        self.p_lapse_dist = Bernoulli(self.lapse_rate)
        super().__init__(**kwargs, validate_args=validate_args)

    @property
    def mean(self):
        return (
            self.lapse_rate * self.lapse_dist.mean
            + (1 - self.lapse_rate) * self.base_dist.mean
        )

    def log_prob(self, rts):
        # rt whose p=0 will have logp=nan, replace with -1000 which will exp() to 0 anyway
        # in logsumexp
        rt_logp = torch.nan_to_num(self.base_dist.log_prob(rts), nan=-1000)
        lapse_logp = self.lapse_dist.log_prob(rts)

        [*batch_shape, rt_shape] = rt_logp.shape
        assert rt_shape == lapse_logp.shape[0]
        lapse_logp = lapse_logp.expand(*batch_shape, -1)

        mix_logps = torch.stack(
            (
                lapse_logp + torch.log(self.lapse_rate),
                rt_logp + torch.log(1 - self.lapse_rate),
            ),
            dim=-1,
        )
        return torch.logsumexp(mix_logps, dim=-1)

    def sample(self, sample_shape=torch.Size([])): # noqa B008
        rt_samps = self.base_dist.sample(sample_shape=sample_shape)
        unif_samps = self.lapse_dist.rsample(sample_shape=rt_samps.shape)
        coinflips = self.p_lapse_dist.sample(sample_shape=rt_samps.shape).int()[..., 0]
        return torch.where(coinflips == 1, unif_samps, rt_samps)


class ExGaussian(Distribution):
    def __init__(self, mean, stddev, lam, validate_args=False, *args, **kwargs):
        self.mean = mean
        self.stddev = stddev
        self.lam = lam

        super().__init__(**kwargs, validate_args=validate_args)

    def log_prob(self, x):
        """
        Same as PyMC
        """
        res = torch.where(
            self.lam > 0.05 * self.stddev,
            -torch.log(self.lam)
            + (self.mean - x) / self.lam
            + 0.5 * (self.stddev / self.lam) ** 2
            + torch.log(
                Normal(loc=self.mean + (self.stddev**2) / self.lam, scale=self.stddev**2).cdf(x)
            ),
            LogNormal(loc=self.mean, scale=self.stddev**2).log_prob(x),
        )
        return res

    def rsample(self, sample_shape=torch.Size()): # noqa B008
        return Normal(loc=self.mean, scale=self.stddev).rsample(
            sample_shape=sample_shape
        ) + Exponential(rate=self.lam).rsample(sample_shape=sample_shape)


class ShiftedGamma(TransformedDistribution):
    r"""
    Creates a shifted log-normal distribution parameterized by
    :attr:`shift` and, :attr:`loc`, and :attr:`scale` where::

    """
    arg_constraints = {
        "concentration": constraints.Positive(),
        "rate": constraints.Positive(),
    }
    support = constraints.Positive()
    has_rsample = True

    def __init__(
        self, shift, concentration, rate, validate_args=False, **kwargs
    ):
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        self.shift = shift
        super().__init__(
            base_dist,
            [AffineTransform(loc=shift, scale=torch.tensor(1.0))],
            validate_args=validate_args,
            **kwargs,
        )

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate

    def log_prob(self, X):
        return torch.where(X < self.shift, torch.nan, super().log_prob(X))


class ShiftedInverseGamma(TransformedDistribution):
    r"""
    Creates a shifted log-normal distribution parameterized by
    :attr:`shift` and, :attr:`loc`, and :attr:`scale` where::

    """
    arg_constraints = {
        "concentration": constraints.Positive(),
        "rate": constraints.Positive(),
    }
    support = constraints.Positive()
    has_rsample = True

    def __init__(
        self, shift, concentration, rate, validate_args=False, **kwargs
    ):
        base_dist = Gamma(concentration, rate, validate_args=validate_args)
        self.shift = shift
        super(ShiftedInverseGamma, self).__init__(
            base_dist,
            [
                PowerTransform(exponent=torch.tensor(-1.0)),
                AffineTransform(loc=shift, scale=torch.tensor(1.0)),
            ],
            validate_args=validate_args,
            **kwargs,
        )

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate


class ShiftedLognormal(TransformedDistribution):
    r"""
    Creates a shifted log-normal distribution parameterized by
    :attr:`shift` and, :attr:`loc`, and :attr:`scale` where::

    """
    arg_constraints = {
        "scale": constraints.Positive(),
    }
    support = constraints.Positive()
    has_rsample = True

    def __init__(self, shift, loc, scale, validate_args=False):
        base_dist = Normal(loc, scale, validate_args=validate_args)
        self.shift = shift
        super(ShiftedLognormal, self).__init__(
            base_dist,
            [ExpTransform(), AffineTransform(loc=shift, scale=torch.tensor(1.0))],
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(ShiftedLognormal, _instance)
        return super(ShiftedLognormal, self).expand(batch_shape, _instance=new)

    @property
    def loc(self):
        return self.base_dist.loc

    @property
    def scale(self):
        return self.base_dist.scale

    @property
    def mean(self):
        return (self.loc + self.scale.pow(2) / 2).exp() + self.shift

    @property
    def variance(self):
        return (self.scale.pow(2).exp() - 1) * (2 * self.loc + self.scale.pow(2)).exp()


def coth(x):
    # probably not numerically terrific
    return torch.cosh(x) / torch.sinh(x)


def csch(x):
    # probably not numerically terrific
    return 1 / torch.sinh(x)

class DDMMomentMatchDistribution(Distribution):
    """
    Distribution over [choices, rts]. However, since rts are always positive, we use the sign of the RTs
    to track choice information (so rt>0 means yes choice, rt<0 means no choice) which means as far as gpytorch knows,
    we still have a univariate outcome.
    There's basically 2 steps here:
        1. Use the DDM parameters to compute moments of the conditional RT distributions
    and choice probabilities using the expressions in https://mae.princeton.edu/sites/default/files/SrivastHSimen-JMatPsy16.pdf.
        This is what the base class does.
        2. Moment-match the moments to some nicer distribution, and pretend that's our likelihood. That
        is what subclasses do.
    """

    SMALL_DRIFT_CUTOFF = (
        1e-2  # use this as cutoff to use asymptotic drift -> 0 expressions
    )
    arg_constraints = {
        "threshold": constraints.Positive(),
        "relative_x0": constraints.Interval(0.2, 0.8),
        "t0": constraints.Positive(),
    }
    support = constraints.Positive()

    def __init__(
        self, drift, threshold, relative_x0, t0, restrict_skew=False, max_shift=None
    ):

        self.drift = drift
        self.threshold = threshold
        self.max_shift = max_shift

        # relative x0 is scaled 0 to 1, x0 is -thresh to thresh
        # boundarySep = thresh * 2
        # relativeInitCond = (x0+z) / boundarySep
        # boundarySep * relativeInitCond = x0+z

        self.x0 = threshold * (2 * relative_x0 - 1)

        kz = drift * threshold
        kx = drift * self.x0

        near_zero_drift = drift.abs() < self.SMALL_DRIFT_CUTOFF

        # as abs(drift) -> 0, use different expressions (expr 30 and 32)
        rt_mean_yes0 = (4 * threshold**2 - (threshold + self.x0) ** 2) / 3
        rt_mean_no0 = (4 * threshold**2 - (threshold - self.x0) ** 2) / 3
        rt_var_yes0 = (32 * threshold**4 - 2 * (threshold + self.x0) ** 4) / 45
        rt_var_no0 = (32 * threshold**4 - 2 * (threshold - self.x0) ** 4) / 45

        # for nonzero drift, expr 29 and 31
        self.rt_mean_yes = (
            torch.where(
                near_zero_drift,
                rt_mean_yes0,
                drift ** (-2) * ((2 * kz * coth(2 * kz)) - (kx + kz) * coth(kx + kz)),
            )
            + t0
        )
        self.rt_mean_no = (
            torch.where(
                near_zero_drift,
                rt_mean_no0,
                drift ** (-2) * ((2 * kz * coth(2 * kz)) - (-kx + kz) * coth(-kx + kz)),
            )
            + t0
        )
        self.rt_var_yes = torch.where(
            near_zero_drift,
            rt_var_yes0,
            drift ** (-4)
            * (
                4 * kz**2 * csch(2 * kz) ** 2
                + 2 * kz * coth(2 * kz)
                - (kx + kz) ** 2 * csch(kx + kz) ** 2
                - (kx + kz) * coth(kx + kz)
            ),
        )
        self.rt_var_no = torch.where(
            near_zero_drift,
            rt_var_no0,
            drift ** (-4)
            * (
                4 * kz**2 * csch(2 * kz) ** 2
                + 2 * kz * coth(2 * kz)
                - (-kx + kz) ** 2 * csch(-kx + kz) ** 2
                - (-kx + kz) * coth(-kx + kz)
            ),
        )

        # expr 36
        rt_3rd_moment_yes = drift ** (-6) * (
            12 * kz**2 * csch(2 * kz) ** 2
            + 16 * kz**3 * coth(2 * kz) * csch(2 * kz) ** 2
            + 6 * kz * coth(2 * kz)
            - 3 * (kz + kx) ** 2 * csch(kx + kz) ** 2
            - 2 * (kx + kz) ** 3 * coth(kz + kx) * csch(kz + kx) ** 2
            - 3 * (kx + kz) * coth(kx + kz)
        )
        rt_3rd_moment_no = drift ** (-6) * (
            12 * kz**2 * csch(2 * kz) ** 2
            + 16 * kz**3 * coth(2 * kz) * csch(2 * kz) ** 2
            + 6 * kz * coth(2 * kz)
            - 3 * (kz - kx) ** 2 * csch(kz - kx) ** 2
            - 2 * (-kx + kz) ** 3 * coth(kz - kx) * csch(kz - kx) ** 2
            - 3 * (-kx + kz) * coth(-kx + kz)
        )
        rt_skew_yes = rt_3rd_moment_yes / self.rt_var_yes ** (3 / 2)
        rt_skew_no = rt_3rd_moment_no / self.rt_var_no ** (3 / 2)

        # expr 37
        # np.sqrt(45/2) = 4.743416490252569
        SQRT45_2 = 4.743416490252569
        rt_skew_yes0 = SQRT45_2 * (
            (8 * (64 * threshold**6 - (threshold + self.x0) ** 6))
            / (21 * (16 * threshold**4 - (threshold + self.x0) ** 4) ** (3 / 2))
        )
        rt_skew_no0 = SQRT45_2 * (
            (8 * (64 * threshold**6 - (threshold - self.x0) ** 6))
            / (21 * (16 * threshold**4 - (threshold - self.x0) ** 4) ** (3 / 2))
        )

        self.rt_skew_yes = torch.where(near_zero_drift, rt_skew_yes0, rt_skew_yes)
        self.rt_skew_no = torch.where(near_zero_drift, rt_skew_no0, rt_skew_no)

        # expr 6 and 9
        self.response_prob = torch.where(
            near_zero_drift,
            (threshold - self.x0) / (2 * threshold),
            1
            - (torch.exp(-2 * kx) - torch.exp(-2 * kz))
            / (torch.exp(2 * kz) - torch.exp(-2 * kz)),
        )

        # these will fail if numerical stability is bad, clamp them
        self.response_prob = self.response_prob.clamp(min=1e-5, max=1 - 1e-5)
        self.rt_var_yes = self.rt_var_yes.clamp(min=1e-5)
        self.rt_var_no = self.rt_var_no.clamp(min=1e-5)
        self.rt_mean_yes = self.rt_mean_yes.clamp(min=1e-5)
        self.rt_mean_no = self.rt_mean_no.clamp(min=1e-5)
        if restrict_skew:
            self.rt_skew_yes = self.rt_skew_yes.clamp(min=0.01, max=10)
            self.rt_skew_no = self.rt_skew_no.clamp(min=0.01, max=10)

        self._make_moment_matched_likelihood()

    def _make_moment_matched_likelihood(self):
        raise NotImplementedError

    @property
    def mean(self):
        return (
            self.response_prob * self.rt_mean_yes
            + (1 - self.response_prob) * self.rt_mean_no
        )

    def rsample(self, sample_shape=torch.Size()): # noqa B008
        choices = self.choice_dist.sample(sample_shape=sample_shape)
        rt_yes = self.rt_yes_dist.rsample(sample_shape=sample_shape)
        rt_no = self.rt_no_dist.rsample(sample_shape=sample_shape)
        return torch.where(choices > 0, rt_yes, -rt_no)

    def log_prob(self, signed_rts):
        # log p(rt, choice | theta) =log p(rt|choice, theta) + log p(choice | theta)
        # p(rt|choice) is our conditional lognormal, p(choice) is bernoulli.
        choices = signed_rts > 0
        yes_log_probs = self.rt_yes_dist.log_prob(torch.abs(signed_rts))
        no_log_probs = self.rt_no_dist.log_prob(torch.abs(signed_rts))
        rt_log_probs = torch.where(choices, yes_log_probs, no_log_probs)

        return rt_log_probs + self.choice_dist.log_prob(choices.float())


class LogNormalDDMDistribution(DDMMomentMatchDistribution):
    def _make_moment_matched_likelihood(self):

        # moment match to lognormal (from https://en.wikipedia.org/wiki/Log-normal_distribution)
        lognormal_mu_yes = torch.log(
            self.rt_mean_yes / torch.sqrt(self.rt_var_yes / self.rt_mean_yes**2 + 1)
        )
        lognormal_sigma_yes = torch.sqrt(
            torch.log(self.rt_var_yes / self.rt_mean_yes**2 + 1)
        )
        lognormal_mu_no = torch.log(
            self.rt_mean_no / torch.sqrt(self.rt_var_no / self.rt_mean_no**2 + 1)
        )
        lognormal_sigma_no = torch.sqrt(
            torch.log(self.rt_var_no / self.rt_mean_no**2 + 1)
        )

        assert (lognormal_sigma_yes > 0.0).all(), lognormal_sigma_yes.min()
        assert (lognormal_sigma_no > 0.0).all(), lognormal_sigma_no.min()

        self.choice_dist = torch.distributions.Bernoulli(probs=self.response_prob)
        self.rt_yes_dist = torch.distributions.LogNormal(
            loc=lognormal_mu_yes, scale=lognormal_sigma_yes
        )
        self.rt_no_dist = torch.distributions.LogNormal(
            loc=lognormal_mu_no, scale=lognormal_sigma_no
        )


class ShiftedLogNormalDDMDistribution(DDMMomentMatchDistribution):
    def _make_moment_matched_likelihood(self):

        # moment match to shifted lognormal (from https://jod.pm-research.com/content/21/4/103,
        # doi:10.3905/jod.2014.21.4.103, lemma 8

        B_yes = 0.5 * (
            self.rt_skew_yes.square()
            + 2
            - torch.sqrt(self.rt_skew_yes**4 + 4 * self.rt_skew_yes.square())
        )

        shifted_lognormal_shift_yes = self.rt_mean_yes - (
            self.rt_var_yes.sqrt() / self.rt_skew_yes
        ) * (1 + B_yes ** (1 / 3) + B_yes ** (-1 / 3))

        shifted_lognormal_var_yes = torch.log(
            1
            + self.rt_var_yes / ((self.rt_mean_yes - shifted_lognormal_shift_yes) ** 2)
        )
        shifted_lognormal_mean_yes = (
            torch.log(self.rt_mean_yes - shifted_lognormal_shift_yes)
            - shifted_lognormal_var_yes**2 / 2
        )

        B_no = 0.5 * (
            self.rt_skew_no.square()
            + 2
            - torch.sqrt(self.rt_skew_no**4 + 4 * self.rt_skew_no.square())
        )

        shifted_lognormal_shift_no = self.rt_mean_no - (
            self.rt_var_no.sqrt() / self.rt_skew_no
        ) * (1 + B_no ** (1 / 3) + B_no ** (-1 / 3))

        shifted_lognormal_var_no = torch.log(
            1 + self.rt_var_no / ((self.rt_mean_no - shifted_lognormal_shift_no) ** 2)
        )
        shifted_lognormal_mean_no = (
            torch.log(self.rt_mean_no - shifted_lognormal_shift_no)
            - shifted_lognormal_var_no**2 / 2
        )

        self.choice_dist = torch.distributions.Bernoulli(probs=self.response_prob)

        shifted_lognormal_shift_yes = shifted_lognormal_shift_yes.clamp(
            min=0, max=self.max_shift
        )
        shifted_lognormal_shift_no = shifted_lognormal_shift_no.clamp(
            min=0, max=self.max_shift
        )

        self.rt_yes_dist = ShiftedLognormal(
            shift=shifted_lognormal_shift_yes,
            loc=shifted_lognormal_mean_yes,
            scale=shifted_lognormal_var_yes.sqrt(),
        )
        self.rt_no_dist = ShiftedLognormal(
            shift=shifted_lognormal_shift_no,
            loc=shifted_lognormal_mean_no,
            scale=shifted_lognormal_var_no.sqrt(),
        )


class ExGaussianDDMDistribution(DDMMomentMatchDistribution):
    def _make_moment_matched_likelihood(self):
        # moment match to exgaussian (from https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution#Parameter_estimation)

        # exgaussian is restricted to skew <= 2, so we clamp

        clamped_yes_skew = torch.clamp(self.rt_skew_yes, max=torch.tensor(2.0))
        clamped_no_skew = torch.clamp(self.rt_skew_no, max=torch.tensor(2.0))
        tau_yes = self.rt_var_yes.sqrt() * (clamped_yes_skew / 2) ** (1 / 3)
        mu_yes = self.rt_mean_yes - tau_yes
        var_yes = self.rt_var_yes * (1 - (clamped_yes_skew / 2) ** (2 / 3))

        tau_no = self.rt_var_no.sqrt() * (clamped_no_skew / 2) ** (1 / 3)
        mu_no = self.rt_mean_no - tau_no
        var_no = self.rt_var_no * (1 - (clamped_no_skew / 2) ** (2 / 3))

        self.choice_dist = torch.distributions.Bernoulli(probs=self.response_prob)
        self.rt_yes_dist = ExGaussian(m=mu_yes, s=var_yes.sqrt(), l=1 / tau_yes)
        self.rt_no_dist = ExGaussian(m=mu_no, s=var_no.sqrt(), l=1 / tau_no)


class ShiftedGammaDDMDistribution(DDMMomentMatchDistribution):
    def _make_moment_matched_likelihood(self):
        # doi:10.1016/j.insmatheco.2020.12.002, 
        # section 4.2.

        a_yes = 4 / self.rt_skew_yes**2
        scale_yes = (self.rt_var_yes / a_yes).sqrt()
        shift_yes = self.rt_mean_yes - a_yes * scale_yes

        a_no = 4 / self.rt_skew_no**2
        scale_no = (self.rt_var_no / a_no).sqrt()
        shift_no = self.rt_mean_no - a_no * scale_no

        shift_yes = shift_yes.clamp(min=0, max=self.max_shift)
        shift_no = shift_no.clamp(min=0, max=self.max_shift)

        self.choice_dist = torch.distributions.Bernoulli(probs=self.response_prob)
        self.rt_yes_dist = ShiftedGamma(
            shift=shift_yes, concentration=a_yes, rate=1 / scale_yes
        )
        self.rt_no_dist = ShiftedGamma(
            shift=shift_no, concentration=a_no, rate=1 / scale_no
        )


class ShiftedInverseGammaDDMDistribution(DDMMomentMatchDistribution):
    def _make_moment_matched_likelihood(self):
        # doi:10.1016/j.insmatheco.2020.12.002, 
        # section 4.3. 

        shift_yes = self.rt_mean_yes - self.rt_var_yes.sqrt() / self.rt_skew_yes * (
            2 + (4 + self.rt_skew_yes.square()).sqrt()
        )
        a_yes = 2 + (self.rt_mean_yes - shift_yes).square() / self.rt_var_yes
        b_yes = (self.rt_mean_yes - shift_yes) * (a_yes - 1)

        shift_no = self.rt_mean_no - self.rt_var_no.sqrt() / self.rt_skew_no * (
            2 + (4 + self.rt_skew_no.square()).sqrt()
        )
        a_no = 2 + (self.rt_mean_no - shift_no).square() / self.rt_var_no
        b_no = (self.rt_mean_no - shift_no) * (a_no - 1)

        shift_yes = shift_yes.clamp(min=0, max=self.max_shift)
        shift_no = shift_no.clamp(min=0, max=self.max_shift)

        self.choice_dist = torch.distributions.Bernoulli(probs=self.response_prob)
        self.rt_yes_dist = ShiftedInverseGamma(
            shift=shift_yes, concentration=a_yes, rate=b_yes
        )
        self.rt_no_dist = ShiftedInverseGamma(
            shift=shift_no, concentration=a_no, rate=b_no
        )


class DDMDistribution(Distribution):

    arg_constraints = {
        "z": constraints.Positive(),
        "relative_x0": constraints.Interval(0.0, 1.0),
        "t0": constraints.Positive(),
    }
    def __init__(self, a, z, relative_x0, t0, eps=1e-10, validate_args=True):

        self.a, self.z, self.relative_x0, self.t0 = broadcast_all(a, z, relative_x0, t0)

        if (
            isinstance(a, Number)
            and isinstance(z, Number)
            and isinstance(relative_x0, Number)
            and isinstance(t0, Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()

        self.eps = eps
        super().__init__(batch_shape=batch_shape, validate_args=validate_args)

    def _standardized_WFPT_large_time(self, t, w, nterms):
        # large time expansion from navarro & fuss

        piSqOv2 = 4.93480220054
        # use nterms that's max over the batch. This guarantees
        # we'll hit our target precision and enable batched
        # computation, but will incur extra cost for the extra
        # terms if not needed.
        k = torch.arange(1, nterms + 1)
        k = k.expand(*w.shape, *k.shape)  # match batch shape to params
        w = w[:, None]  # broadcast an extra dim for w we can reduce sum over

        terms = (
            torch.pi
            * k
            * torch.exp(-(k**2) * t * piSqOv2)
            * torch.sin(k * torch.pi * w)
        )
        assert terms.shape == (*t.shape[:-1], *self.batch_shape, nterms)
        return terms.sum(-1)

    def _standardized_WFPT_small_time(self, t, w, nterms):
        # small time expansion navarro & fuss

        fr = math.floor(-(nterms - 1) / 2)
        to = math.ceil((nterms - 2) // 2)
        k = torch.arange(fr, to + 1)
        k = k.expand(*w.shape, *k.shape)
        w = w[:, None]  # broadcast an extra dim for w we can reduce sum over

        terms = (
            1
            / torch.sqrt(2 * torch.pi * t**3)
            * (w + 2 * k)
            * torch.exp(-((w + 2 * k) ** 2) / (2 * t))
        )
        assert terms.shape == (*t.shape[:-1], *self.batch_shape, nterms)
        return terms.sum(0)

    def log_prob(self, signed_rt):
        """
        Log probability of first passage time of double-threshold wiener process
        (aka "pure DDM" of Bogacz et al.). Uses series truncation of Navarro & Fuss 2009
        """

        shifted_t = signed_rt.abs() - self.t0  # correct for the shift
        # normalize time (this also implicitly broadcasts)
        normT = shifted_t / (self.relative_x0**2)

        # if t is below NDT, return -inf
        t_below_ndt = normT <= 0

        # by default return hit of lower bound, so if resp is correct flip
        # signflip based on choice as needed
        driftsign = torch.where(signed_rt > 0, -1, 1)
        a = self.a * driftsign
        relative_x0 = torch.where(signed_rt > 0, 1 - self.relative_x0, self.relative_x0)

        largeK = torch.ceil(
            torch.sqrt(
                (-2 * torch.log(torch.pi * normT * self.eps)) / (torch.pi**2 * normT)
            )
        )
        smallK = torch.ceil(
            2
            + torch.sqrt(
                -2 * normT * torch.log(2 * self.eps * torch.sqrt(2 * torch.pi * normT))
            )
        )

        # if eps is too big for bound to be valid, adjust
        smallK[self.eps > (1 / (2 * torch.sqrt(2 * torch.pi * normT)))] = 2
        bound_invalid = self.eps > (1 / (torch.pi * torch.sqrt(normT)))
        largeK[bound_invalid] = torch.ceil(
            (1 / (torch.pi * torch.sqrt(normT[bound_invalid])))
        )

        # pick the smaller of large and small k options, then
        # take the max so we can batch properly without needing ragged arrays
        nterms = torch.min(largeK, smallK)[torch.logical_not(t_below_ndt)]
        if nterms.max() - nterms.min() > 100:
            warnings.warn(
                "Number of series terms over a batch varies by more than 100, compute costs may be increased",
                RuntimeWarning,
                stacklevel=2
            )

        nterms = nterms.max()

        use_large_time = largeK >= smallK

        prob = torch.zeros_like(normT)
        prob[t_below_ndt] = -torch.inf

        large_time_approx = self._standardized_WFPT_large_time(normT, relative_x0, nterms)
        small_time_approx = self._standardized_WFPT_small_time(normT, relative_x0, nterms)
        prob[use_large_time] = large_time_approx[use_large_time.squeeze()]
        prob[torch.logical_not(use_large_time)] = small_time_approx[
            torch.logical_not(use_large_time).squeeze()
        ]

        boundarySep = 2 * self.z

        # scale from the std case to whatever is our actual
        scaler = (1 / relative_x0**2) * torch.exp(
            -a * boundarySep * relative_x0 - (a**2 * shifted_t / 2)
        )

        return torch.log(scaler * prob)
