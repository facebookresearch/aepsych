#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import time
from typing import Dict, List, Optional

from aepsych.utils_logging import getLogger
from ax.core.search_space import SearchSpaceDigest
from ax.core.types import TCandidateMetadata
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.fit import fit_gpytorch_mll
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor

logger = getLogger()


class AEPsychSurrogate(Surrogate):
    def __init__(self, max_fit_time: Optional[float] = None, **kwargs) -> None:
        self.max_fit_time = max_fit_time
        super().__init__(**kwargs)

    def fit(
        self,
        datasets: List[SupervisedDataset],
        metric_names: List[str],
        search_space_digest: SearchSpaceDigest,
        candidate_metadata: Optional[List[List[TCandidateMetadata]]] = None,
        state_dict: Optional[Dict[str, Tensor]] = None,
        refit: bool = True,
        **kwargs,
    ) -> None:
        self.construct(
            datasets=datasets,
            metric_names=metric_names,
            **dataclasses.asdict(search_space_digest),
        )
        self._outcomes = metric_names
        if state_dict:
            self.model.load_state_dict(state_dict)

        if state_dict is None or refit:
            mll = self.mll_class(self.model.likelihood, self.model, **self.mll_options)
            optimizer_kwargs = {}
            if self.max_fit_time is not None:
                # figure out how long evaluating a single samp
                starttime = time.time()
                _ = mll(self.model(datasets[0].X()), datasets[0].Y().squeeze())
                single_eval_time = time.time() - starttime
                n_eval = int(self.max_fit_time / single_eval_time)
                logger.info(f"fit maxfun is {n_eval}")
                optimizer_kwargs["options"] = {"maxfun": n_eval}

            logger.info("Starting fit...")
            starttime = time.time()
            fit_gpytorch_mll(
                mll, optimizer_kwargs=optimizer_kwargs
            )  # TODO: Support flexible optimizers
            logger.info(f"Fit done, time={time.time()-starttime}")
