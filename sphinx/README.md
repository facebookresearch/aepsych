# Sphinx API reference

This file describes the sphinx setup for auto-generating the AEPsych API reference.


## Installation

**Requirements**:
- sphinx >= 3.0  (Install via `pip install sphinx`)


## Building

From the `aepsych/sphinx` directory, run `make html`.

Generated HTML output can be found in the `aepsych/sphinx/_build` directory. The main index page is: `aepsych/sphinx/_build/html/index.html`

Note: You can delete the content in the `_build` directory by running `make clean`.


## Structure

`sphinx/index.rst` contains the main index. The API reference for each module lives in its own file, e.g. `models.rst` for the `aepsych.models` module.
