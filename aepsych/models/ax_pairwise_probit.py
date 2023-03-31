from __future__ import annotations
from botorch.models.pairwise_gp import PairwiseGP
import torch
from aepsych.config import Config
from botorch.models.likelihoods.pairwise import (
    PairwiseProbitLikelihood,
)
from aepsych.models.base import AEPsychModel
from aepsych.utils import promote_0d
from scipy.stats import norm
from aepsych.models.utils import select_inducing_points
from typing import Optional, Union, Type, Tuple
from aepsych.factory import default_mean_covar_factory
from aepsych.config import Config
from aepsych.utils_logging import getLogger
from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.variational import (
    _VariationalDistribution,
    _VariationalStrategy,
    VariationalStrategy,
)
from botorch.models.utils.inducing_point_allocators import (
    InducingPointAllocator,
)
from gpytorch.means import Mean
from typing import Optional
from aepsych.utils_logging import getLogger
from torch import Tensor


logger = getLogger()


class PairwiseGPModel(AEPsychModel, PairwiseGP):
    name = "PairwiseProbitModel"
    outcome_type = "binary"
    stimuli_per_trial = 1

    def __init__(
        self,
        train_X: Tensor,
        train_Y: Optional[Tensor] = None,
        likelihood: Optional[Likelihood] = None,
        num_outputs: int = 1,
        learn_inducing_points: bool = True,
        covar_module: Optional[Kernel] = None,
        mean_module: Optional[Mean] = None,
        variational_distribution: Optional[_VariationalDistribution] = None,
        variational_strategy: Type[_VariationalStrategy] = VariationalStrategy,
        inducing_points: Optional[Union[Tensor, int]] = None,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        inducing_point_allocator: Optional[InducingPointAllocator] = None,
    ) -> None:
        self.outcome_transform = outcome_transform
        self.input_transform = input_transform
        dim = self.dim
        bounds = torch.stack((self.lb, self.ub))
        input_transform = Normalize(d=self.dim, bounds=bounds)
        if covar_module is None:
            config = Config(
                config_dict={
                    "default_mean_covar_factory": {
                        "lb": str(self.lb.tolist()),
                        "ub": str(self.ub.tolist()),
                    }
                }
            )  # type: ignore
            _, covar_module = default_mean_covar_factory(config)

        super().__init__(
            datapoints=None,
            comparisons=None,
            covar_module=covar_module,
            jitter=1e-3,
            input_transform=input_transform,
        )

        self.covar_module = covar_module
        self.botorch_model_class = None
        self.dim = dim  # The Pairwise constructor sets self.dim = None.

    def _pairs_to_comparisons(
        self, x: torch.tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        comparisons = []
        for i in range(x.shape[0]):
            for j in range(i + 1, x.shape[0]):
                comparisons.append((i, j) if y[i] > y[j] else (j, i))
        return x, torch.LongTensor(comparisons)

    @classmethod
    def get_mll_class(cls):
        return PairwiseLaplaceMarginalLogLikelihood

    def sample(self, x, num_samples, rereference="x_min"):
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        if rereference is None:
            return self.posterior(x).rsample(torch.Size([num_samples]))

        if rereference == "x_min":
            x_ref = self.lb
        elif rereference == "x_max":
            x_ref = self.ub
        elif rereference == "f_max":
            x_ref = torch.Tensor(self.get_max()[1])
        elif rereference == "f_min":
            x_ref = torch.Tensor(self.get_min()[1])
        else:
            raise RuntimeError(
                f"Unknown rereference type {rereference}! Options: x_min, x_max, f_min, f_max."
            )

        x_stack = torch.vstack([x, x_ref])
        samps = self.posterior(x_stack).rsample(torch.Size([num_samples]))
        samps, samps_ref = torch.split(samps, [samps.shape[1] - 1, 1], dim=1)
        if rereference == "x_min" or rereference == "f_min":
            return samps - samps_ref
        else:
            return -samps + samps_ref

    def predict(
        self, x, probability_space=False, num_samples=1000, rereference="x_min"
    ):
        if rereference is not None:
            samps = self.sample(x, num_samples, rereference)
            fmean, fvar = samps.mean(0).squeeze(), samps.var(0).squeeze()
        else:
            post = self.posterior(x)
            fmean, fvar = post.mean.squeeze(), post.variance.squeeze()

        if probability_space:
            return (
                promote_0d(norm.cdf(fmean)),
                promote_0d(norm.cdf(fvar)),
            )
        else:
            return fmean, fvar

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
    def get_config_options(cls, config: Config, name: str = None):
        options = super().get_config_options(config, name)
        classname = cls.__class__.__name__

        inducing_point_method = config.get(
            classname, "inducing_point_method", fallback="auto"
        )
        inducing_size = config.getint(classname, "inducing_size", fallback=10)
        learn_inducing_points = config.getboolean(
            classname, "learn_inducing_points", fallback=False
        )
        cls.lb = config.gettensor(classname, "lb")
        cls.ub = config.gettensor(classname, "ub")
        cls.dim = cls.lb.shape[0]

        options.update(
            {
                "inducing_size": inducing_size,
                "inducing_point_method": inducing_point_method,
                "learn_inducing_points": learn_inducing_points,
                "likelihood": PairwiseProbitLikelihood(),
            }
        )
        return options
