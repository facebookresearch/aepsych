import gpytorch
import numpy as np
import torch
from aepsych.acquisition import MCLevelSetEstimation
from aepsych.acquisition.objective import ProbitObjective
from aepsych.modelbridge.base import ModelBridge, _prune_extra_acqf_args
from aepsych.models import GPClassificationModel
from botorch.fit import fit_gpytorch_model
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples


class SingleProbitModelbridge(ModelBridge):

    outcome_type = "single_probit"

    def __init__(
        self,
        lb,
        ub,
        restarts=10,
        samps=1000,
        dim=1,
        acqf=None,
        extra_acqf_args=None,
        model=None,
    ):
        if extra_acqf_args is None:
            extra_acqf_args = {}

        super().__init__(
            lb=lb, ub=ub, dim=dim, acqf=acqf, extra_acqf_args=extra_acqf_args
        )

        self.restarts = restarts
        self.samps = samps

        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
        if model is None:
            self.model = GPClassificationModel(
                inducing_min=self.lb, inducing_max=self.ub
            )
        else:
            self.model = model

    def fit(self, train_x, train_y):
        n = train_y.shape[0]
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, n)
        self.model.train()
        self.model.set_train_data(train_x, train_y)
        fit_gpytorch_model(self.mll)

    def gen(self, num_points=1, **kwargs):

        self.model.eval()
        train_x = self.model.train_inputs[0]
        acq = self._get_acquisition_fn()

        new_candidate, batch_acq_values = optimize_acqf(
            acq_function=acq,
            bounds=torch.tensor(np.c_[self.lb, self.ub]).T.to(train_x),
            q=num_points,
            num_restarts=self.restarts,
            raw_samples=self.samps,
        )

        return new_candidate.numpy()

    def predict(self, x):
        post = self.model.posterior(x)
        return post.mean.squeeze(), post.variance.squeeze()

    def sample(self, x, num_samples):
        return self.model(x).rsample(torch.Size([num_samples]))

    @classmethod
    def from_config(cls, config):

        classname = cls.__name__
        model = GPClassificationModel.from_config(config)

        lb = config.gettensor(classname, "lb")
        ub = config.gettensor(classname, "ub")
        restarts = config.getint(classname, "restarts", fallback=10)
        samps = config.getint(classname, "samps", fallback=1000)
        assert lb.shape[0] == ub.shape[0], "bounds are of different shapes!"
        dim = lb.shape[0]

        acqf = config.getobj("experiment", "acqf", fallback=MCLevelSetEstimation)
        acqf_name = acqf.__name__

        default_extra_acqf_args = {
            "beta": 3.98,
            "target": 0.75,
            "objective": ProbitObjective,
        }
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
        return cls(
            lb=lb,
            ub=ub,
            restarts=restarts,
            samps=samps,
            dim=dim,
            acqf=acqf,
            model=model,
            extra_acqf_args=extra_acqf_args,
        )


class SingleProbitModelbridgeWithSongHeuristic(SingleProbitModelbridge):
    def gen(self, num_points: int = 1, noise_scale=0.2, **kwargs):

        # Generate the points at which to sample
        X = draw_sobol_samples(
            bounds=torch.Tensor(np.c_[self.lb, self.ub]).T, n=self.samps, q=1
        ).squeeze(1)

        # Draw n samples
        f_samp = self.sample(X, num_samples=1000)
        acq = self._get_acquisition_fn()
        acq_vals = acq.acquisition(acq.objective(f_samp))
        # normalize
        acq_vals = acq_vals - acq_vals.min()
        acq_vals = acq_vals / acq_vals.max()
        # add noise
        acq_vals = acq_vals + torch.randn_like(acq_vals) * noise_scale

        # Find the point closest to target
        best_vals, best_indx = torch.topk(acq_vals, k=num_points)
        return X[best_indx]
