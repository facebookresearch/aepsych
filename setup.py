#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup

REQUIRES = [
    "numpy<2.0, >=1.20",
    "matplotlib",
    "torch",
    "scipy==1.14.1",
    "SQLAlchemy==1.4.46",
    "dill",
    "pandas",
    "aepsych_client==0.3.0",
    "statsmodels",
    "botorch==0.12.0",
]

BENCHMARK_REQUIRES = ["tqdm", "pathos", "multiprocess"]

DEV_REQUIRES = BENCHMARK_REQUIRES + [
    "coverage",
    "flake8",
    "sqlalchemy-stubs",  # for mypy stubs
    "mypy==1.14.0",
    "parameterized",
    "scikit-learn",  # used in unit tests
]

try:
    with open("requirements-fmt.txt") as fh:
        deps = fh.read()
        DEV_REQUIRES += [dep for dep in deps.split("\n") if not dep.startswith("#")]

except FileNotFoundError:
    # Fallback versions for formatting
    DEV_REQUIRES += [
        "black==24.4.2",
        "ruff-api==0.1.0",
        "stdlibs==2024.1.28",
        "ufmt==2.8.0",
        "usort==1.0.8.post1",
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
    python_requires=">=3.10",
    packages=find_packages(),
    description="Adaptive experimetation for psychophysics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "benchmark": BENCHMARK_REQUIRES,
    },
    entry_points={
        "console_scripts": [
            "aepsych_server = aepsych.server.server:main",
            "aepsych_database = aepsych.server.utils:main",
        ],
    },
)
