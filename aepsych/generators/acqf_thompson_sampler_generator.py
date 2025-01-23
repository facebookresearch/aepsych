#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
import torch
from aepsych.models.base import ModelProtocol
from aepsych.utils_logging import getLogger
from numpy.random import choice

from .grid_eval_acqf_generator import GridEvalAcqfGenerator

logger = getLogger()


class AcqfThompsonSamplerGenerator(GridEvalAcqfGenerator):
    """Generator that samples points in a grid with probability proportional to an acquisition function value."""

    def _gen(
        self,
        num_points: int,
        model: ModelProtocol,
        fixed_features: Optional[Dict[int, float]] = None,
        **gen_options,
    ) -> torch.Tensor:
        """
        Generates the next query points by optimizing the acquisition function.

        Args:
            num_points (int): The number of points to query.
            model (ModelProtocol): The fitted model used to evaluate the acquisition function.
            fixed_features: (Dict[int, float], optional): Parameters that are fixed to specific values.
            gen_options (dict): Additional options for generating points, including:
                - "seed": Random seed for reproducibility.

        Returns:
            torch.Tensor: Next set of points to evaluate, with shape [num_points x dim].
        """
        logger.info("Starting gen...")
        starttime = time.time()

        grid, acqf_vals = self._eval_acqf(
            self.samps, model, fixed_features, **gen_options
        )
        acqf_vals -= acqf_vals.min()
        probability_dist = acqf_vals / acqf_vals.sum()
        candidate_idx = choice(
            np.arange(acqf_vals.shape[0]),
            size=num_points,
            p=probability_dist.detach().numpy(),
        )
        new_candidate = grid[candidate_idx]

        logger.info(f"Gen done, time={time.time()-starttime}")
        return new_candidate
