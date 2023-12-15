#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import numpy as np
import torch
from aepsych.models.base import AEPsychModel
from botorch.models import ModelListGP


class AEPsychModelListGP(AEPsychModel, ModelListGP):
    def fit(self):
        for model in self.models:
            model.fit()

    def predict_probability(
        self, x: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Query the model for posterior mean and variance in probability space.
        This method works by calling `predict_probability` separately for each model
        in self.models. If a model does not implement "predict_probability", it will
        instead return `model.predict`.

        Args:
            x (torch.Tensor): Points at which to predict from the model.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Posterior mean and variance at queries points.
        """
        prob_list = []
        vars_list = []
        for model in self.models:
            if hasattr(model, "predict_probability"):
                prob, var = model.predict_probability(x)
            else:
                prob, var = model.predict(x)
            prob_list.append(prob.unsqueeze(-1))
            vars_list.append(var.unsqueeze(-1))
        probs = torch.hstack(prob_list)
        vars = torch.hstack(vars_list)

        return probs, vars

    @classmethod
    def get_mll_class(cls):
        return None
