#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup

root_dir = os.path.dirname(__file__)

REQUIRES = [
    "matplotlib",
    "torch",
    "pyzmq==19.0.2",
    "scipy",
    "sklearn",
    "gpytorch>=1.4",
    "botorch>=0.6.1",
    "SQLAlchemy",
    "dill",
    "pandas",
    "tqdm",
    "pathos",
]

DEV_REQUIRES = [
    "coverage",
    "flake8",
    "black",
    "numpy>=1.20, " "sqlalchemy-stubs",  # for mypy stubs
    "mypy",
    "parameterized",
]

with open(os.path.join(root_dir, "README.md"), "r") as fh:
    long_description = fh.read()

setup(
    name="aepsych",
    version="0.1.0",
    python_requires=">=3.8",
    packages=find_packages(),
    description="Adaptive experimetation for psychophysics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=REQUIRES,
    entry_points={
        "console_scripts": [
            "aepsych_server = aepsych.server.server:main",
        ],
    },
)

extras_require = (
    {
        "dev": DEV_REQUIRES,
    },
)
