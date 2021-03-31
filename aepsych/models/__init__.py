import sys

from ..config import Config
from .gp_classification import GPClassificationModel
from .monotonic_rejection_gp import (
    MonotonicRejectionGP,
    MonotonicGPLSE,
    MonotonicGPLSETS,
)

__all__ = [
    "GPClassificationModel",
    "MonotonicRejectionGP",
    "MonotonicGPLSE",
    "MonotonicGPLSETS",
]

Config.register_module(sys.modules[__name__])
