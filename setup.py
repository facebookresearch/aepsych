#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from os import path

from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "Readme.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="aepsych",
    version="0.1",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
)
