The AEPsych website was created with [Docusaurus](https://docusaurus.io/).
FontAwesome icons were used under the
[Creative Commons Attribution 4.0 International](https://fontawesome.com/license).

# Installation

**Requirements**:
- You need [Node](https://nodejs.org/en/) >= 8.x,
[Yarn](https://yarnpkg.com/en/) >= 1.5, and [Sphix](https://www.sphinx-doc.org/en/master/usage/installation.html) in order to build the  website.


To Install AEPsych on your local machine please clone the project repository.


```bash
 git clone https://github.com/facebookresearch/aepsych.git
```

Switch into to the `website/` directory from within the AEPsych project root and install dependencies before starting the server:
```bash
cd website
npm install
npm start
```
The website will open on http://localhost:3000.
Please use the link above it does not open automatically.

This website is built with react.
Therefore, any changes to the content will auto-update in your browser.




# Building Documentation

Please Note: The installation instructions above do not re-build the API reference (auto-generated by Sphinx) or parse and embed the tutorial notebooks.

To rebuild the website with updated content run the following script from the root directory of the AEPsych project.


1) Give your self permision to execute `build_docs.sh`
```bash
 chmod +x ./scripts/build_docs.sh
```
2) Run the following command from the root directory
```bash
./scripts/build_docs.sh
```

# Publishing Website (gh-pages)

The site is hosted on GitHub pages, via the `gh-pages` branch of the AEPsych
[GitHub repo](https://github.com/facebookresearch/aepsych/tree/gh-pages).

To publish the website with updated content run the following script from the root directory of the AEPsych project.

1) Give your self permision to execute `publish_site.sh`
```bash
 chmod +x ./scripts/publish_site.sh
```

2) Run the following command from the root directory
```bash
./scripts/publish_site.sh
```
