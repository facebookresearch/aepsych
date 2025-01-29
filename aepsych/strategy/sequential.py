#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from typing import Any, List, Optional, Union

import numpy as np
import torch
from aepsych.config import Config
from aepsych.strategy import Strategy
from aepsych.utils_logging import getLogger

logger = getLogger()


class SequentialStrategy(object):
    """Runs a sequence of strategies defined by its config

    All getter methods defer to the current strat

    Args:
        strat_list (list[Strategy]): TODO make this nicely typed / doc'd
    """

    def __init__(self, strat_list: List[Strategy]) -> None:
        """Initialize the SequentialStrategy object.

        Args:
            strat_list (List[Strategy]): The list of strategies.
        """
        self.strat_list = strat_list
        self._strat_idx = 0
        self._suggest_count = 0
        self.x: Optional[torch.Tensor]
        self.y: Optional[torch.Tensor]

    @property
    def _strat(self) -> Strategy:
        """Get the current strategy.

        Returns:
            Strategy: The current strategy.
        """
        return self.strat_list[self._strat_idx]

    def __getattr__(self, name: str) -> Any:
        """Get the attribute of the current strategy.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute of the current strategy.
        """
        # return current strategy's attr if it's not a container attr
        if "strat_list" not in vars(self):
            raise AttributeError("Have no strategies in container, what happened?")
        return getattr(self._strat, name)

    def _make_next_strat(self) -> None:
        """Switch to the next strategy."""
        if (self._strat_idx + 1) >= len(self.strat_list):
            warnings.warn(
                "Ran out of generators, staying on final generator!", RuntimeWarning
            )
            return

        # populate new model with final data from last model
        assert (
            self.x is not None and self.y is not None
        ), "Cannot initialize next strategy; no data has been given!"
        self.strat_list[self._strat_idx + 1].add_data(self.x, self.y)

        self._suggest_count = 0
        self._strat_idx = self._strat_idx + 1

    def gen(self, num_points: int = 1, **kwargs) -> torch.Tensor:
        """Generate the next set of points to evaluate.

        Args:
            num_points (int): The number of points to generate. Defaults to 1.

        Returns:
            torch.Tensor: The next set of points to evaluate.
        """
        if self._strat.finished:
            self._make_next_strat()
        self._suggest_count = self._suggest_count + num_points

        return self._strat.gen(num_points=num_points, **kwargs)

    def finish(self) -> None:
        """Finish the strategy."""
        self._strat.finish()

    @property
    def finished(self) -> bool:
        """Check if the strategy is finished.

        Returns:
            bool: True if the strategy is finished, False otherwise.
        """
        return self._strat_idx == (len(self.strat_list) - 1) and self._strat.finished

    def add_data(
        self, x: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Add new data points to the strategy.

        Args:
            x (Union[np.ndarray, torch.Tensor]): The input data points.
            y (Union[np.ndarray, torch.Tensor]): The output data points.
        """
        self._strat.add_data(x, y)

    @classmethod
    def from_config(cls, config: Config) -> SequentialStrategy:
        """Create a SequentialStrategy object from a configuration object.

        Args:
            config (Config): The configuration object.

        Returns:
            SequentialStrategy: The SequentialStrategy object.
        """
        strat_names = config.getlist("common", "strategy_names", element_type=str)

        # ensure strat_names are unique
        assert len(strat_names) == len(
            set(strat_names)
        ), f"Strategy names {strat_names} are not all unique!"

        strats = []
        for name in strat_names:
            strat = Strategy.from_config(config, str(name))
            strats.append(strat)

        return cls(strat_list=strats)
