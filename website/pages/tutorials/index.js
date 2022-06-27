/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
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
              Lorem ipsum dolor sit amet, consectetur adipiscing elit,
              sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
              </p>

              {/* <p>
                <h4>Why AEPsych?</h4>
                  <p>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit,
                    ed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                  </p>
                <h4>Header</h4>
                  <p>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit,
                    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                  </p>
                <h4>Header</h4>
                  <p>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit,
                    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                  </p>
                <h4>Header</h4>
                  <p>
                    Lorem ipsum dolor sit amet, consectetur adipiscing elit,
                    sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
                  </p>
              </p> */}
            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = TutorialHome;
