import sys

from ..config import Config
from .monotonic import MonotonicSingleProbitModelbridge
from .single_probit import (
    SingleProbitModelbridge,
    SingleProbitModelbridgeWithSongHeuristic,
)

__all__ = [
    "MonotonicSingleProbitModelbridge",
    "SingleProbitModelbridge",
    "SingleProbitModelbridgeWithSongHeuristic",
]

Config.register_module(sys.modules[__name__])
