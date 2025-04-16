#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "Readme.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(os.path.join("aepsych_client", "version.py"), "r") as fh:
    for line in fh.readlines():
        if line.startswith("__version__"):
            version = line.split('"')[1]

setup(
    name="aepsych_client",
    version=version,
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
)
