import sys

from . import (
    acquisition,
    config,
    factory,
    benchmark,
    modelbridge,
    models,
    strategy,
    utils,
)
from .config import Config
from .models import GPClassificationModel
from .strategy import (
    EpsilonGreedyModelWrapperStrategy,
    ModelWrapperStrategy,
    SequentialStrategy,
    SobolStrategy,
)

__all__ = [
    # modules
    "acquisition",
    "benchmark",
    "config",
    "factory",
    "modelbridge",
    "models",
    "strategy",
    "utils",
    # classes
    "EpsilonGreedyModelWrapperStrategy",
    "GPClassificationModel",
    "ModelWrapperStrategy",
    "SequentialStrategy",
    "SobolStrategy",
]

Config.register_module(sys.modules[__name__])
