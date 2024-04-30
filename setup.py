#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup

REQUIRES = [
    "matplotlib",
    "torch",
    "scipy",
    "SQLAlchemy==1.4.46",
    "dill",
    "pandas",
    "aepsych_client==0.3.0",
    "statsmodels",
    "ax-platform==0.3.7",
]

BENCHMARK_REQUIRES = ["tqdm", "pathos", "multiprocess"]

DEV_REQUIRES = BENCHMARK_REQUIRES + [
    "coverage",
    "flake8",
    "black",
    "numpy>=1.20",
    "sqlalchemy-stubs",  # for mypy stubs
    "mypy",
    "parameterized",
    "scikit-learn",  # used in unit tests
]

VISUALIZER_REQUIRES = [
    "voila==0.3.6",
    "ipywidgets==7.6.5",
]

with open("Readme.md", "r") as fh:
    long_description = fh.read()

with open(os.path.join("aepsych", "version.py"), "r") as fh:
    for line in fh.readlines():
        if line.startswith("__version__"):
            version = line.split('"')[1]


setup(
    name="aepsych",
    version=version,
    python_requires=">=3.8",
    packages=find_packages(),
    description="Adaptive experimetation for psychophysics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "benchmark": BENCHMARK_REQUIRES,
        "visualizer": VISUALIZER_REQUIRES,
    },
    entry_points={
        "console_scripts": [
            "aepsych_server = aepsych.server.server:main",
            "aepsych_database = aepsych.server.utils:main",
        ],
    },
)
