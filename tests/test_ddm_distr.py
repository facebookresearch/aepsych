#!/usr/bin/env python3
import unittest

import numpy as np

import torch
from functools import partial
from torch.func import grad
from torch import vmap
from numbers import Number

from aepsych.distributions import DDMMomentMatchDistribution


class TestDDMDistr(DDMMomentMatchDistribution):
    def _make_moment_matched_likelihood(self):
        pass

global_atol = 1e-3

def ddm_mgf(alpha, drift, x0, threshold, response=1):
    """
    Moment-generating function of the Wiener First Passage time distribution (DDM)
    """
    if response == 0:
        drift = -drift.clone()
        threshold = -threshold.clone()
    return torch.exp(drift * (threshold - x0)) * (
        torch.sinh((threshold + x0) * torch.sqrt(drift**2 - 2 * alpha))
        / torch.sinh(2 * threshold * torch.sqrt(drift**2 - 2 * alpha))
    )


def ddm_cgf(alpha, drift, x0, threshold, response=1):
    """
    Cumulant-generating function of the Wiener First Passage time distribution (DDM)
    """

    return torch.log(ddm_mgf(alpha, drift, x0, threshold, response=response))


def ddm_moment_cumulant(n, drift, x0, threshold, fun="cumulant", response=1):
    """
    Function to generate arbitrary moments or cumulants of DDM by autodiff,
    vectorized over drift (but not other arguments currently, TODO).
    """
    assert fun in ("moment", "cumulant")
    if isinstance(drift, Number):
        drift = torch.Tensor([drift])
    if fun == "moment":
        deriv_fun = ddm_mgf
    elif fun == "cumulant":
        deriv_fun = ddm_cgf
    else:
        raise RuntimeError(f"fun should be moment or cumulant, got {fun}")
    for _ in range(n):
        deriv_fun = grad(deriv_fun)
    moment_fun = partial(deriv_fun, torch.tensor(0.0), response=response)

    moment_fun_vmap = vmap(moment_fun, in_dims=(0, None, None))

    return moment_fun_vmap(drift, x0, threshold)



class DDMMomemtnMatchTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)
        torch.manual_seed(1)
        self.f = torch.randn(100)
        # things are numerically unstable as drift -> 0. 
        # in the momentmatch expressions we use limiting expressions
        # if drift is too small but we don't have that in moment/cumulant
        # so exclude from tests. TODO: can probably improve numerical stability.
        self.f = self.f + torch.sign(self.f) * 0.05
        self.relative_x0 = torch.tensor(0.1)
        self.t0 = torch.tensor(0.15)
        self.rt_dist = TestDDMDistr(
            drift=self.f,
            threshold=torch.tensor(0.5),
            relative_x0=self.relative_x0,
            t0=self.t0,
        )
        self.x0 = torch.tensor(0.5 * (2 * self.relative_x0- 1))

    def test_mean(self):
        # sanity check mean
        expected_yes_mean = ddm_moment_cumulant(
            n=1,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=1,
        )
        expected_no_mean = ddm_moment_cumulant(
            n=1,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=0,
        )
        self.assertTrue(
            torch.allclose(
                self.rt_dist.rt_mean_yes,
                expected_yes_mean + self.t0,
                atol=global_atol,
            )
        )
        self.assertTrue(
            torch.allclose(
                self.rt_dist.rt_mean_no,
                expected_no_mean + self.t0,
                atol=global_atol,
            )
        )

    def test_var(self):
        # sanity check var
        expected_yes_var = ddm_moment_cumulant(
            n=2,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=1,
        )
        expected_no_var = ddm_moment_cumulant(
            n=2,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=0,
        )
        self.assertTrue(
            torch.allclose(self.rt_dist.rt_var_yes, expected_yes_var, atol=global_atol)
        )
        self.assertTrue(
            torch.allclose(self.rt_dist.rt_var_no, expected_no_var, atol=global_atol)
        )

    def test_skew(self):
        # sanity check skew
        expected_yes_var = ddm_moment_cumulant(
            n=2,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=1,
        )
        expected_no_var = ddm_moment_cumulant(
            n=2,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=0,
        )
        expected_yes_skew = ddm_moment_cumulant(
            n=3,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=1,
        ) / (expected_yes_var ** (3 / 2))
        expected_no_skew = ddm_moment_cumulant(
            n=3,
            drift=self.rt_dist.drift,
            x0=self.x0,
            threshold=self.rt_dist.threshold,
            response=0,
        ) / (expected_no_var ** (3 / 2))
        self.assertTrue(
            torch.allclose(
                self.rt_dist.rt_skew_yes, expected_yes_skew, atol=global_atol
            )
        )
        self.assertTrue(
            torch.allclose(self.rt_dist.rt_skew_no, expected_no_skew, atol=global_atol)
        )


if __name__ == "__main__":
    unittest.main()
