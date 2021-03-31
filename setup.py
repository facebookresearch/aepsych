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
