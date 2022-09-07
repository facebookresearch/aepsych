/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the scripts directory.
 *
 * @format
 */

const React = require('react');

const CWD = process.cwd();

const CompLibrary = require(`${CWD}/node_modules/docusaurus/lib/core/CompLibrary.js`);
const Container = CompLibrary.Container;
const MarkdownBlock = CompLibrary.MarkdownBlock;

const TutorialSidebar = require(`${CWD}/core/TutorialSidebar.js`);

class TutorialHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <TutorialSidebar currentTutorialID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">AEPsych Tutorials</h1>
            </header>
            <body>
              <p>
                The tutorials here are designed to help you get familiar with the parts of
                AEPsych relevant to you, whether from a psychophysics or CSML perspective.
                Additional tutorials will be available soon.
              </p>

              {}
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
