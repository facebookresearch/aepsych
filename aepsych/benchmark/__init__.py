from .benchmark import Benchmark, combine_benchmarks
from .logger import BenchmarkLogger
from .problem import Problem, LSEProblem
from .test_functions import (
    make_songetal_testfun,
    novel_detection_testfun,
    novel_discrimination_testfun,
)

__all__ = [
    "combine_benchmarks",
    "Benchmark",
    "DaskBenchmark",
    "BenchmarkLogger",
    "PathosBenchmark",
    "Problem",
    "LSEProblem",
    "make_songetal_testfun",
    "novel_detection_testfun",
    "novel_discrimination_testfun",
]

try:
    from .pathos_benchmark import PathosBenchmark
    __all__.append("PathosBenchmark")
except ImportError:
    pass  # For systems without pathos
