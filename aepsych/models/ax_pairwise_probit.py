from __future__ import annotations
import os

from botorch.models.pairwise_gp import PairwiseGP
import torch
from aepsych.config import Config
from botorch.models.likelihoods.pairwise import (
    PairwiseProbitLikelihood,
    PairwiseLikelihood,
)
from aepsych.models.base import AEPsychModel
from sklearn.datasets import make_classification
from aepsych.utils import get_dim, promote_0d, _process_bounds
from scipy.stats import norm
from aepsych.models.utils import select_inducing_points
from aepsych.models.variational_gp import BinaryClassificationGP

import time, gpytorch, numpy as np
from typing import Any, Dict, Optional, Union, Type, Tuple
from aepsych.factory import default_mean_covar_factory
from aepsych.config import Config
from aepsych.utils_logging import getLogger
from botorch.fit import fit_gpytorch_mll
from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.kernels import Kernel
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from gpytorch.variational import (
    _VariationalDistribution,
    _VariationalStrategy,
    CholeskyVariationalDistribution,
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

from botorch.models.utils.inducing_point_allocators import (
    GreedyVarianceReduction,
    InducingPointAllocator,
)

from gpytorch.means import ConstantMean, Mean
import dataclasses
import time
from typing import Dict, List, Optional
from aepsych.utils_logging import getLogger
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from botorch.fit import fit_gpytorch_mll
from botorch.utils.datasets import SupervisedDataset
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

        datapoints, comparisons = self._pairs_to_comparisons(train_X, train_Y)
        print("Datapoints: ", datapoints, " Comparisons: ", comparisons)
        super().__init__(
            datapoints=datapoints,
            comparisons=None,
            covar_module=covar_module,
            jitter=1e-3,
            input_transform=input_transform,
        )

        self.covar_module = covar_module
        self.botorch_model_class = None
        self.dim = dim  # The Pairwise constructor sets self.dim = None.
        self.datapoints = datapoints

    def _pairs_to_comparisons(
        self, x: torch.tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        """
        Takes x, y structured as pairs and judgments and
        returns pairs and comparisons as PairwiseGP requires
        """
        # This needs to take a unique over the feature dim by flattening
        # over pairs but not instances/batches. This is actually tensor
        # matricization over the feature dimension but awkward in numpy
        unique_coords = torch.unique(
            torch.transpose(x, 1, 0).reshape(self.dim, -1), dim=1
        )                                    

        def _get_index_of_equal_row(arr, x: torch.Tensor, dim=0):
            # print("Inputed X: ", x.unsqueeze(-1))
            # print("Inputed arr: ", arr)
            torch_eq = torch.eq(arr, x).reshape(1, -1)
            # print("Torch.eq: ", torch_eq)
            # print("Torch.all: ", torch.all(torch_eq, dim=dim))
            item = torch.all(torch_eq, dim=dim).nonzero().item()
            # print("Item: ", item)
            return item

        comparisons = []
        for pair, judgement in zip(x, y):
            # print("Pair: ", pair)
            comparison = (
                _get_index_of_equal_row(unique_coords[0, ...], pair[..., 0]),
                _get_index_of_equal_row(unique_coords[1, ...], pair[..., 1]),
            )
            if judgement == 0:
                comparisons.append(comparison)
            else:
                comparisons.append(comparison[::-1])
        return unique_coords.T, torch.LongTensor(comparisons)

    @classmethod
    def get_mll_class(cls):
        return PairwiseLaplaceMarginalLogLikelihood

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

    # def fit(
    #     self,
    #     datasets: List[SupervisedDataset],
    #     metric_names: List[str],
    #     search_space_digest: SearchSpaceDigest,
    #     candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
    #     state_dict: Optional[Dict[str, Tensor]] = None,
    #     refit: bool = True,
    #     **kwargs,
    # ) -> None:
    #     print("Fitting PairwiseGPModel ..................")
    #     self.train()
    #     self._outcomes = metric_names
    #     if state_dict:
    #         self.model.load_state_dict(state_dict)

    #     if state_dict is None or refit:
    #         mll = self.get_mll_class()(self.likelihood, self)
    #         optimizer_kwargs = {}
    #         if self.max_fit_time is not None:
    #             # figure out how long evaluating a single samp
    #             starttime = time.time()

    #             if isinstance(self, PairwiseGPModel):
    #                 datapoints, comparisons = self._pairs_to_comparisons(
    #                     datasets[0].X(), datasets[0].Y().squeeze()
    #                 )
    #                 self.set_train_data(datapoints, comparisons)
    #                 _ = mll(self.model(datapoints), comparisons)
    #             else:
    #                 _ = mll(self.model(datasets[0].X()), datasets[0].Y().squeeze())
    #             single_eval_time = time.time() - starttime
    #             n_eval = int(self.max_fit_time / single_eval_time)
    #             logger.info(f"fit maxfun is {n_eval}")
    #             optimizer_kwargs["options"] = {"maxfun": n_eval}

    #         logger.info("Starting fit...")
    #         print(self.max_fit_time)
    #         starttime = time.time()
    #         fit_gpytorch_mll(
    #             mll, optimizer_kwargs=optimizer_kwargs
    #         )  # TODO: Support flexible optimizers
    #         logger.info(f"Fit done, time={time.time()-starttime}")

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


# a = PairwiseGPModel().train_inputs
# b = BinaryClassificationGP().forward()
