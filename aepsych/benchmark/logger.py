#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Optional, List, Mapping

import pandas as pd


class BenchmarkLogger:
    """Logger for recording benchmark outputs."""

    def __init__(self, log_every: Optional[int] = None):
        """Initialize benchmark logger

        Args:
            log_every (int, optional): Logging interval during an experiment.
                Defaults to only logging at the end.
        """
        self._log: List[Dict[str, object]] = []
        self.log_every = log_every

    def log_at(self, i: int) -> bool:
        """Check if we should log on this trial index.

        Args:
            i (int): Trial index to (maybe) log at.

        Returns:
            bool: True if this trial should be logged.
        """
        if self.log_every is not None:
            return i % self.log_every == 0
        else:
            return False

    def log(
        self,
        flatconfig: Mapping[str, object],
        metrics: Mapping[str, object],
        trial_id: int,
        elapsed: float,
        rep: int,
        final: bool = False,
    ) -> None:
        """Log trial data.

        Args:
            flatconfig (Mapping[str, object]): Flattened configuration for this benchmark.
            metrics (Mapping[str, object]): Metrics to log.
            trial_id (int): Current trial index.
            elapsed (float): Duration of this trial.
            rep (int): Repetition index of this trial.
            final (bool, optional): Mark this as the final trial in a run? Defaults to False.
        """
        out: Dict[str, object] = {"elapsed": elapsed, "trial_id": trial_id, "rep": rep, "final": final}
        out.update(flatconfig)
        out.update(metrics)
        self._log.append(out)

    def pandas(self) -> pd.DataFrame:
        return pd.DataFrame(self._log)
