/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const React = require('react');

const CompLibrary = require('../../core/CompLibrary.js');

const MarkdownBlock = CompLibrary.MarkdownBlock; /* Used to read markdown */
const Container = CompLibrary.Container;
const GridBlock = CompLibrary.GridBlock;

const bash = (...args) => `~~~bash\n${String.raw(...args)}\n~~~`;

class HomeSplash extends React.Component {
  render() {
    const {siteConfig, language = ''} = this.props;
    const {baseUrl, docsUrl} = siteConfig;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    const langPart = `${language ? `${language}/` : ''}`;
    const docUrl = (doc) => `${baseUrl}${docsPart}${langPart}${doc}`;

    const SplashContainer = (props) => (
      <div className="homeContainer">
        <div className="homeSplashFade">
          <div className="wrapper homeWrapper">{props.children}</div>
        </div>
      </div>
    );

    const Logo = (props) => (
      // <div className="projectLogo">
        <img className="splash-logo" src={props.img_src} alt="Project Logo" />
      // </div>
    );

    const ProjectTitle = (props) => (
      <h2 className="projectTitle">
        {props.title}
        <small>{props.tagline}</small>
      </h2>
    );

    const PromoSection = (props) => (
      <div className="section promoSection">
        <div className="promoRow">
          <div className="pluginRowBlock">{props.children}</div>
        </div>
      </div>
    );

    const Button = (props) => (
      <div className="pluginWrapper buttonWrapper">
        <a className="button" href={props.href} target={props.target}>
          {props.children}
        </a>
      </div>
    );

    return (
      <SplashContainer>
            {/* Main logo */}
           {/* <Logo img_src={`${baseUrl}img/placeholder.png`} /> */}
        <div className="inner">
          <ProjectTitle tagline={siteConfig.tagline} title={siteConfig.title} />
          <PromoSection>
            <Button class="splash-btns" href={docUrl("introduction")}>INTRODUCTION</Button>
            <Button class="splash-btns" href={'#quickstart'}>Get Started</Button>

            {/* Tutorial link will go here --- NEED TO UPDATE */}
            <Button class="splash-btns" href={`${baseUrl}tutorials/`}>Tutorials</Button>
          </PromoSection>
        </div>
      </SplashContainer>
    );
  }
}

class Index extends React.Component {
  render() {
    const {config: siteConfig, language = ''} = this.props;
    const {baseUrl} = siteConfig;

    const Block = (props) => (
      <Container
        padding={['bottom', 'top']}
        id={props.id}
        background={props.background}>
        <GridBlock
          align="center"
          contents={props.children}
          layout={props.layout}
        />
      </Container>
    );


    // getStartedSection
    const pre = '```';
    // For 1) pip install
    const firstCodeSection = `${pre}python
    cd aepsych
    pip install -r requirements.txt
    pip install -e .
    `;

    // For 2) Usage - first
    const secondCodeSection = `${pre}python
    python aepsych/server/server.py
    `;
    // For 3) Usage - second
    const thirdCodeSection  = `${pre}python
    {
     "type":<TYPE>,
     "version":<VERSION>,
     "message":<MESSAGE>,
    }
    `;
    // For 4) Setup
    const forthCodeSection = `${pre}python
    {
    "type":"setup",
    "version":"0.01",
    "message":{"config_str":<PASTED CONFIG STRING>}
    }
    `;


    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <div className="start-info-splash">
        </div>
        <Container>
        <div className="start-info-div">
          <p className="disable start-info">
        AEPsych supports python 3.8+. We recommend
        installing AEPsych under a virtual environment like Anaconda.
        Created a virtual environment for AEPsych once it is activated
        you can install our requirements and then install AEPsych.
        </p>
        </div>
          <ol>
            <li>
              <h4>Install AEPsych:</h4>
              <p className="disable">via pip</p>
              <MarkdownBlock>{firstCodeSection}</MarkdownBlock>
            </li>
            <li>
              <h4>Usage:</h4>
              <p className="disable">The canonical way of using AEPsych is to launch it in server mode.</p>
              <MarkdownBlock>{secondCodeSection}</MarkdownBlock>
              <p className="disable">(you can call python aepsych/server/server.py --help to see
              additional arguments). The server accepts messages over either p
              unix socket or ZMQ, and all messages are formatted using JSON. All
              messages have the following format:
              </p>
              <MarkdownBlock>{thirdCodeSection}</MarkdownBlock>
              <p className="disable">
              Version can be omitted, in which case we default to the
              oldest / unversioned handler for this message type. There are
              five message types: setup, resume, ask, tell and exit.
              </p>
            </li>
            <li>
              <h4>Setup:</h4>
              <p className="disable">
              The setup message prepares the server for making
              suggestions and accepting data. The setup message can be
              formatted as either INI or a python dict (similar to JSON)
              format, and an example for psychometric threshold estimation
              is given in configs/single_lse_example.ini. It looks like this:
              </p>
              <MarkdownBlock>{forthCodeSection}</MarkdownBlock>
              <a href="https://github.com/facebookresearch/aepsych">More examples</a>
            </li>

          </ol>
        </Container>
      </div>
    );


    const Features = () => (
      <div className="productShowcaseSection" style={{textAlign: 'center'}}>
      <h2>Key Features</h2>
      <Block layout="threeColumn">
          {[
            {
              content:
                'Text goes here...',
              image: `${baseUrl}img/puzzle.svg`,
              imageAlign: 'top',
              title: 'Header',
            },
            {
              content:
                'Text goes here...',
              image: `${baseUrl}img/arrow-up-right.svg`,
              imageAlign: 'top',
              title: 'Header',
            },
            {
              content:
              'Text goes here...',
              image: `${baseUrl}img/blocks.svg`,
              imageAlign: 'top',
              title: 'Header',
            },
          ]}
        </Block>
      </div>
    );

// May not need for V1 of site
    // const Reference = () => (
    //   <div
    //     className="productShowcaseSection"
    //     id="reference"
    //     style={{textAlign: 'center'}}>
    //     <h2>References</h2>
    //     <Container>
    //      <p>
    //         Lorem Ipsum is simply dummy text of the printing and typesetting industry.
    //         Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
    //         when an unknown printer took a galley of type and scrambled it to make a type
    //         specimen book.
    //       </p>
    //     </Container>
    //   </div>
    // );

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Features />
          {/* <Reference /> */}
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
