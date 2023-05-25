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

const DemoSidebar = require(`${CWD}/core/DemoSidebar.js`);

const DemoButton = ({ imageUrl, demoUrl, buttonText }) => (
  <a href={demoUrl} style={{ textDecoration: 'none' }}>
    <div style={{ display: 'inline-block', width: '40%', margin: "10px 12px", position: "relative" }}>
      <img src={imageUrl} alt="Demo Image" style={{ width: '100%', height: 'auto', }} />
      <a className="demo-btns" href={demoUrl} style={{ position: 'absolute', top: '10%', left: '50%', transform: 'translate(-50%, -50%)', color: "white", border: "none", fontWeight: "900", fontSize: "1.5rem" }}>
        {buttonText}
      </a>
    </div>
  </a>
);


class DemoHome extends React.Component {
  render() {
    return (
      <div className="docMainWrapper wrapper">
        <DemoSidebar currentDemoID={null} />
        <Container className="mainContainer documentContainer postContainer">
          <div className="post">
            <header className="postHeader">
              <h1 className="postHeaderTitle">AEPsych Unity Demos</h1>
            </header>
            <body >
              <div style={{ display: 'flex', flexWrap: 'wrap' }}>
                <DemoButton
                  imageUrl={`${this.props.config.baseUrl}img/particle-effect-demo.png`}
                  demoUrl="/demos/ParticleEffectDemo"
                  buttonText="Particle Demo"
                />

              </div>

            </body>
          </div>
        </Container>
      </div>
    );
  }
}

module.exports = DemoHome;
