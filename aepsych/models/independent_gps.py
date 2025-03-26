#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from typing import Any, Mapping

import torch
from aepsych.config import Config
from aepsych.models.base import AEPsychModelMixin
from botorch.posteriors import PosteriorList, TransformedPosterior
from torch.nn import ModuleDict


class IndependentGPsModel(AEPsychModelMixin):
    def __init__(self, model_dict: ModuleDict):
        super().__init__()
        # Proper type-hints for ModuleDicts are not currently supported
        # https://github.com/pytorch/pytorch/issues/80821
        self.models: Mapping[str, AEPsychModelMixin] = model_dict  # type: ignore
        self.model_names = list(self.models.keys())
        self._num_outputs = sum([model._num_outputs for model in model_dict.values()])

    def __getitem__(self, key):
        # Allow both integer and string indices to access model directly
        if isinstance(key, int):
            return self.models[self.model_names[key]]
        elif isinstance(key, str):
            return self.models[str]
        else:
            raise KeyError("Use an int or a str to access submodels.")

    @property
    def num_outputs(self) -> int:
        r"""The number of outputs of the model.

        Equal to the sum of the number of outputs of the individual models
        in the ModelList.
        """
        return sum(self.models[model_name].num_outputs for model_name in self.models)

    @property
    def outcome_type(self) -> list[str]:
        return self.outcome_types

    @property
    def outcome_types(self) -> list[str]:
        return [model.outcome_type for model in self.models.values()]

    @outcome_types.setter
    def outcome_types(self, *args, **kwargs):
        raise AttributeError(
            "outcome_types cannot be set on the IndependentGPsModel, set them individually for each underlying model instead."
        )

    @property
    def train_inputs(self):
        inputs = [model.train_inputs for model in self.models.values()]

        if inputs[0] is None and not all([input is None for input in inputs]):
            raise ValueError("Some models but not all have None as train_inputs")

        if not all(torch.equal(tensor[0], inputs[0][0]) for tensor in inputs[1:]):
            raise ValueError("Models have different train_inputs")

        input = inputs[0][0]

        if input.ndim == 2:
            # Add the q dimension, assumed to be 1
            input = input.unsqueeze(dim=1)

        return (input,)

    @train_inputs.setter
    def train_inputs(self, *args, **kwargs):
        raise AttributeError(
            "train_inputs cannot be set on the IndependentGPsModel, set them individually for each underlying model instead."
        )

    def fit(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
        """Fit underlying models.

        Args:
            train_x (torch.Tensor): (n x dim) inputs.
            train_y (torch.LongTensor): (n x m) responses, where m is the the number of GPs held by this model. GPs are ordered according to the order they were inserted upon initialization
        """
        for i, model_name in enumerate(self.models):
            self.models[model_name].fit(
                train_x=train_x, train_y=train_y[:, i], **kwargs
            )

    def update(self, train_x: torch.Tensor, train_y: torch.Tensor, **kwargs) -> None:
        """Update underlying models.

        Args:
            train_x (torch.Tensor): (n x dim) inputs.
            train_y (torch.LongTensor): (n x m) responses, where m is the the number of GPs held by this model. GPs are ordered according to the order they were inserted upon initialization
        """
        for i, model_name in enumerate(self.models):
            self.models[model_name].update(
                train_x=train_x, train_y=train_y[:, i], **kwargs
            )

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the individual models for posterior means and variances, and concatenate the results.

        Args:
            x (torch.Tensor): Points (n x dim) at which to predict from the model.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at queries points, as (n x m) tensors, where m is the number of GPs held by this model. GPs are ordered according to the order they were inserted upon initialization
        """
        means = []
        vars = []
        for model_name in self.models:
            mean, var = self.models[model_name].predict(x)
            means.append(mean.unsqueeze(-1))
            vars.append(var.unsqueeze(-1))
        return torch.hstack(means), torch.hstack(vars)

    def predict_transform(
        self,
        x: torch.Tensor,
        transformed_posterior_cls: type[TransformedPosterior] | None = None,
        transform_map: dict[str, type[TransformedPosterior]] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Query the individual models for posterior means and variances, transform them, and concatenate the results.

        Args:
            x (torch.Tensor): Points (n x dim) at which to predict from the model.
            transformed_posterior_cls (TransformedPosterior type, optional): Ignored,
                just kept for API uniformity.
            transform_map (dict[str, TransformedPosterior type], optional): A mapping of
                model names and transformed posteriors. For a given model, if it has a
                TransformedPosterior, it will be applied to its outputs.
            **kwargs: Kwargs for transforms, must be the same for all transforms
        Returns:
            tuple[torch.Tensor, torch.Tensor]: Posterior mean and variance at queries points, as (n x m) tensors, where m is the number of GPs held by this model. GPs are ordered according to the order they were inserted upon initialization.
        """
        if transformed_posterior_cls is not None:
            warnings.warn(
                "transformed_posterior_cls is set but will be ignored.", UserWarning
            )
        means = []
        vars = []
        if transform_map is None:
            transform_map = {}
        for model_name in self.models:
            if model_name in transform_map:
                transf = transform_map[model_name]
                mean, var = self.models[model_name].predict_transform(
                    x=x, transformed_posterior_cls=transf, **kwargs
                )
            else:
                mean, var = self.models[model_name].predict_transform(x=x)
            means.append(mean.unsqueeze(-1))
            vars.append(var.unsqueeze(-1))

        return torch.hstack(means), torch.hstack(vars)

    def posterior(self, X: torch.Tensor, posterior_transform=None) -> PosteriorList:
        if X.ndim == 2:
            # Add the q dimension, assumed to be 1
            X = X.unsqueeze(dim=1)

        posteriors = [
            self.models[model_name].posterior(X) for model_name in self.models
        ]
        posterior_list = PosteriorList(*posteriors)

        if posterior_transform is not None:
            return posterior_transform(posterior_list)
        else:
            return posterior_list

    @classmethod
    def get_config_options(
        cls,
        config: Config,
        name: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        options = options or {}
        name = name or cls.__name__

        model_dict = ModuleDict()
        model_names = config.getlist(name, "models", element_type=str)
        for model_name in model_names:
            if model_name in Config.registered_names:
                model_cls = Config.registered_names[model_name]
            else:  # Aliased class
                model_cls = config.getobj(model_name, "class")
            if not hasattr(model_cls, "from_config"):
                raise ValueError(
                    f"IndependentGPsModel was given a model {model_cls} that cannot be configured."
                )
            model = model_cls.from_config(config, model_name)
            model_dict[model_name] = model  # type: ignore

        options.update({"model_dict": model_dict})

        return options
