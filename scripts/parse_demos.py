#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter, MarkdownExporter, PythonExporter

TEMPLATE = """const CWD = process.cwd();

const React = require('react');
const Demo = require(`${{CWD}}/core/Demo.js`);

class DemoPage extends React.Component {{
  render() {{
      const {{config: siteConfig}} = this.props;
      const {{baseUrl}} = siteConfig;
      return <Demo baseUrl={{baseUrl}} demoID="{}"/>;
  }}
}}

module.exports = DemoPage;

"""

JS_SCRIPTS = """
<script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js"></script>
"""


def validate_demo_links(repo_dir: str) -> None:
    """Checks that all .zip files that present are linked on the website, and vice
    versa, that any linked demos has an associated .zip file present.
    """

    with open(os.path.join(repo_dir, "website", "demos.json")) as f:
        demo_config = json.load(f)

    demo_ids = {x["id"] for v in demo_config.values() for x in v}

    demo_names = {
        fn.replace(".zip", "")
        for fn in os.listdir(os.path.join(repo_dir, "demos"))
        if fn[-4:] == ".zip"
    }

    # Check if the ID is present in the set and if both "_Mac" and "_Win" endings exist
    for id in demo_ids:
        if f"{id}_Mac" in demo_names and f"{id}_Win" in demo_names:
            print(f"Both '{id}_Mac' and {id}_Win' demos .zip files are present.")
        else:
            print(f"'{id}_Mac' or {id}_Win' .zip demos are not present.")


def gen_demos(repo_dir: str) -> None:
    """Generate HTML demos for AEPsych Docusaurus site for download."""
    with open(os.path.join(repo_dir, "website", "demos.json"), "r") as f:
        demo_config = json.load(f)

    # create output directories if necessary
    html_out_dir = os.path.join(repo_dir, "website", "_demos")
    files_out_dir = os.path.join(repo_dir, "website", "static", "files", "demos")
    for d in (html_out_dir, files_out_dir):
        if not os.path.exists(d):
            os.makedirs(d)

    demo_ids = {x["id"] for v in demo_config.values() for x in v}

    for d_id in demo_ids:
        print(f"Generating {d_id} demo")

        # convert markdown to HTML
        md_in_path = os.path.join(repo_dir, "demos", "markdown", f"{d_id}.md")
        with open(md_in_path, "r") as infile:
            markdown_content = infile.read()

        notebook_node = nbformat.v4.new_notebook()
        markdown_cell = nbformat.v4.new_markdown_cell(markdown_content)
        notebook_node["cells"] = [markdown_cell]
        exporter = HTMLExporter(template_name="classic")
        html, meta = exporter.from_notebook_node(notebook_node)

        # pull out html div for notebook
        soup = BeautifulSoup(html, "html.parser")
        nb_meat = soup.find("div", {"id": "notebook-container"})
        del nb_meat.attrs["id"]
        nb_meat.attrs["class"] = ["notebook"]
        html_out = JS_SCRIPTS + str(nb_meat)

        # generate html file
        html_out_path = os.path.join(
            html_out_dir,
            f"{d_id}.html",
        )
        with open(html_out_path, "w") as html_outfile:
            html_outfile.write(html_out)

        # generate JS file
        script = TEMPLATE.format(d_id)
        js_out_path = os.path.join(repo_dir, "website", "pages", "demos", f"{d_id}.js")
        with open(js_out_path, "w") as js_outfile:
            js_outfile.write(script)

        # output demo in zip format
        mac_source_path = os.path.join(repo_dir, "demos", f"{d_id}_Mac.zip")
        mac_zip_out_path = os.path.join(files_out_dir, f"{d_id}_Mac.zip")
        shutil.copy(mac_source_path, mac_zip_out_path)

        win_source_path = os.path.join(repo_dir, "demos", f"{d_id}_Win.zip")
        win_zip_out_path = os.path.join(files_out_dir, f"{d_id}_Win.zip")
        shutil.copy(win_source_path, win_zip_out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JS, HTML, and zip files for demos."
    )
    parser.add_argument(
        "-w",
        "--repo_dir",
        metavar="path",
        required=True,
        help="aepsych repo directory.",
    )
    args = parser.parse_args()
    validate_demo_links(args.repo_dir)
    gen_demos(args.repo_dir)
