import sys

from ..config import Config
from .factory import (
    default_mean_covar_factory,
    monotonic_mean_covar_factory,
    song_mean_covar_factory,
)

__all__ = [
    "default_mean_covar_factory",
    "monotonic_mean_covar_factory",
    "song_mean_covar_factory",
]

Config.register_module(sys.modules[__name__])
