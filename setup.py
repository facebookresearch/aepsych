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
    "gpytorch>=1.9.0",
    "botorch>=0.8.0",
    "SQLAlchemy",
    "dill",
    "pandas",
    "tqdm",
    "pathos",
    "aepsych_client",
    "voila==0.3.6",
    "ipywidgets==7.6.5",
    "statsmodels",
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

with open(os.path.join(root_dir, "aepsych", "version.py"), "r") as fh:
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
