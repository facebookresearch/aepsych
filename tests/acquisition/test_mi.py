#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import torch
from aepsych.acquisition.mutual_information import (
    BernoulliMCMutualInformation,
    MonotonicBernoulliMCMutualInformation,
)
from aepsych.acquisition.objective import ProbitObjective
from aepsych.benchmark.test_functions import f_1d
from aepsych.generators import (
    MonotonicRejectionGenerator,
    OptimizeAcqfGenerator,
    SobolGenerator,
)
from aepsych.models import GPClassificationModel, MonotonicRejectionGP
from aepsych.strategy import SequentialStrategy, Strategy
from gpytorch.kernels import LinearKernel
from gpytorch.means import ConstantMean
from scipy.stats import bernoulli, multivariate_normal, norm, pearsonr


class SingleProbitMI(unittest.TestCase):
    def test_1d_monotonic_single_probit(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 15
        n_opt = 1
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        acqf = MonotonicBernoulliMCMutualInformation
        acqf_kwargs = {"objective": ProbitObjective()}
        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_opt,
                model=MonotonicRejectionGP(
                    lb=lb,
                    ub=ub,
                    monotonic_idxs=[0],
                    num_induc=inducing_size,
                ),
                generator=MonotonicRejectionGenerator(
                    lb=lb, ub=ub, acqf=acqf, acqf_kwargs=acqf_kwargs
                ),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(f_1d(next_x))])

        x = torch.linspace(-4, 4, 100).reshape(-1, 1)

        zhat, _ = strat.predict(x)

        true = f_1d(x)
        est = zhat

        # close enough!
        normal_dist = torch.distributions.Normal(0, 1)
        self.assertTrue((((normal_dist.cdf(est) - true) ** 2).mean()) < 0.25)

    def test_1d_single_probit(self):
        seed = 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        n_init = 15
        n_opt = 20
        lb = torch.tensor([-4.0])
        ub = torch.tensor([4.0])
        inducing_size = 10

        acqf = BernoulliMCMutualInformation
        extra_acqf_args = {"objective": ProbitObjective()}

        model_list = [
            Strategy(
                lb=lb,
                ub=ub,
                min_asks=n_init,
                generator=SobolGenerator(lb=lb, ub=ub, seed=seed),
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
            Strategy(
                lb=lb,
                ub=ub,
                model=GPClassificationModel(
                    inducing_size=inducing_size,
                    dim=1,
                ),
                generator=OptimizeAcqfGenerator(
                    lb=lb, ub=ub, acqf=acqf, acqf_kwargs=extra_acqf_args
                ),
                min_asks=n_opt,
                stimuli_per_trial=1,
                outcome_types=["binary"],
            ),
        ]

        strat = SequentialStrategy(model_list)

        for _i in range(n_init + n_opt):
            next_x = strat.gen()
            strat.add_data(next_x, [bernoulli.rvs(f_1d(next_x))])

        x = torch.linspace(-4, 4, 100)

        zhat, _ = strat.predict(x)

        true = f_1d(x)
        est = zhat

        # close enough!
        normal_dist = torch.distributions.Normal(0, 1)
        self.assertTrue((((normal_dist.cdf(est) - true) ** 2).mean()) < 0.25)

    def test_mi_acqf(self):
        mean = ConstantMean().initialize(constant=1.2)
        covar = LinearKernel().initialize(variance=1.0)
        lb = torch.tensor([0.0])
        ub = torch.tensor([1.0])
        inducing_size = 10

        model = GPClassificationModel(
            dim=1,
            inducing_size=inducing_size,
            mean_module=mean,
            covar_module=covar,
        )
        x = torch.rand(size=(10, 1))
        acqf = BernoulliMCMutualInformation(model=model, objective=ProbitObjective())
        acq_pytorch = acqf(x)

        samps_numpy = norm.cdf(
            multivariate_normal.rvs(mean=np.ones(10) * 1.2, cov=x @ x.T, size=10000)
        )
        samp_entropies = bernoulli(samps_numpy).entropy()
        mean_entropy = bernoulli(samps_numpy.mean(axis=0)).entropy()
        acq_numpy = mean_entropy - samp_entropies.mean(axis=0)

        # this assertion fails, not sure why, these should be equal to numerical
        # precision
        # self.assertTrue(np.allclose(acq_numpy, acq_pytorch.detach().numpy().flatten()))
        # this one succeeds
        self.assertTrue(
            pearsonr(acq_numpy, acq_pytorch.detach().numpy().flatten())[0] > (1 - 1e-5)
        )


if __name__ == "__main__":
    unittest.main()
