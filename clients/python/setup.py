#!/usr/bin/env python3
# Copyright 2004-present Facebook. All Rights Reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "Readme.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aepsych_client",
    version="0.3.0",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
)
