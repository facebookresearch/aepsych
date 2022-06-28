#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import os

from bs4 import BeautifulSoup

#The base_url must match the base url in the /website/siteConfig.js
# Note if it is not updated API doc searchbar will not be displayed
# 1) update base_url below
base_url = "/aepsych/"

js_scripts = """
<script type="text/javascript" id="documentation_options" data-url_root="./" src="{0}js/documentation_options.js"></script>
<script type="text/javascript" src="{0}js/jquery.js"></script>
<script type="text/javascript" src="{0}js/underscore.js"></script>
<script type="text/javascript" src="{0}js/doctools.js"></script>
<script type="text/javascript" src="{0}js/language_data.js"></script>
<script type="text/javascript" src="{0}js/searchtools.js"></script>
""".format(base_url)  # noqa: E501

# 2) update
# Search.loadIndex("/<<update to match baseUrl>>/js/searchindex.js"
search_js_scripts = """
  <script type="text/javascript">
    jQuery(function() { Search.loadIndex("/aepsych/js/searchindex.js"); });
  </script>

  <script type="text/javascript" id="searchindexloader"></script>
"""


def parse_sphinx(input_dir, output_dir):
    for cur, _, files in os.walk(input_dir):
        for fname in files:
            if fname.endswith(".html"):
                with open(os.path.join(cur, fname), "r") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    doc = soup.find("div", {"class": "document"})
                    wrapped_doc = doc.wrap(soup.new_tag("div", **{"class": "sphinx"}))
                # add js
                if fname == "search.html":
                    out = js_scripts + search_js_scripts + str(wrapped_doc)
                else:
                    out = js_scripts + str(wrapped_doc)
                output_path = os.path.join(output_dir, os.path.relpath(cur, input_dir))
                os.makedirs(output_path, exist_ok=True)
                with open(os.path.join(output_path, fname), "w") as fout:
                    fout.write(out)

    # update reference in JS file
    with open(os.path.join(input_dir, "_static/searchtools.js"), "r") as js_file:
        js = js_file.read()
    js = js.replace(
        "DOCUMENTATION_OPTIONS.URL_ROOT + '_sources/'", "'_sphinx-sources/'"
    )
    with open(os.path.join(input_dir, "_static/searchtools.js"), "w") as js_file:
        js_file.write(js)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Strip HTML body from Sphinx docs.")
    parser.add_argument(
        "-i",
        "--input_dir",
        metavar="path",
        required=True,
        help="Input directory for Sphinx HTML.",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar="path",
        required=True,
        help="Output directory in Docusaurus.",
    )
    args = parser.parse_args()
    parse_sphinx(args.input_dir, args.output_dir)
