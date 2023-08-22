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
  <a href={demoUrl} className="button-link">
    <div className="demo-btn-div">
      <div className="image-container">
        <img src={imageUrl} alt="Demo Image" />
        <div className="overlay">
          <a className="demo-btn" src={demoUrl}>
            {buttonText}
          </a>
        </div>
      </div>
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
                  demoUrl={`${this.props.config.baseUrl}demos/ParticleEffectDemo`}
                  buttonText="Particle Effect Demo"
                />
                <DemoButton
                  imageUrl={`${this.props.config.baseUrl}img/throw-optimizer-demo.png`}
                  demoUrl={`${this.props.config.baseUrl}demos/ThrowOptimizerDemo`}
                  buttonText="VR Throw Optimizer Demo"
                />
                <DemoButton
                  imageUrl={`${this.props.config.baseUrl}img/yanny-laurel-demo.png`}
                  demoUrl={`${this.props.config.baseUrl}demos/YannyLaurelDemo`}
                  buttonText="Yanny-Laurel Threshold Demo"
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
