#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Callable

from aepsych.utils_logging import getLogger

logger = getLogger()


def ensure_model_is_fresh(f: Callable) -> Callable:
    """Decorator to ensure that the model is up-to-date before running a method.

    Args:
        f (Callable): The function to wrap.

    Returns:
        Callable: The wrapped function.
    """

    def wrapper(self, *args, **kwargs):
        if self.can_fit and not self._model_is_fresh:
            starttime = time.time()
            if self._count % self.refit_every == 0 or self.refit_every == 1:
                logger.info("Starting fitting (no warm start)...")
                # don't warm start
                self.fit()
            else:
                logger.info("Starting fitting (warm start)...")
                # warm start
                self.update()
            logger.info(f"Fitting done, took {time.time()-starttime}")
        self._model_is_fresh = True
        return f(self, *args, **kwargs)

    return wrapper
