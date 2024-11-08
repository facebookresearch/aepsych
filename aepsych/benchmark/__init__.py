#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .benchmark import Benchmark, DerivedValue
from .example_problems import (
    ContrastSensitivity6d,
    DiscrimHighDim,
    DiscrimLowDim,
    Hartmann6Binary,
)
from .pathos_benchmark import PathosBenchmark, run_benchmarks_with_checkpoints
from .problem import LSEProblem, LSEProblemWithEdgeLogging, Problem
from .test_functions import (
    discrim_highdim,
    make_songetal_testfun,
    modified_hartmann6,
    novel_detection_testfun,
    novel_discrimination_testfun,
)

__all__ = [
    "Benchmark",
    "DerivedValue",
    "PathosBenchmark",
    "PathosBenchmark",
    "Problem",
    "LSEProblem",
    "LSEProblemWithEdgeLogging",
    "make_songetal_testfun",
    "novel_detection_testfun",
    "novel_discrimination_testfun",
    "modified_hartmann6",
    "discrim_highdim",
    "run_benchmarks_with_checkpoints",
    "DiscrimLowDim",
    "DiscrimHighDim",
    "Hartmann6Binary",
    "ContrastSensitivity6d",
]
