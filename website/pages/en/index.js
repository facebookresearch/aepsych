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
           <Logo img_src={`${baseUrl}img/placeholder.png`} />
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
    // Example for model fitting
    const modelFitCodeExample = `${pre}python
    <-- code goes here -->
    `;

    // Example for defining an acquisition function
    const constrAcqFuncExample = `${pre}python
    <-- code goes here -->
    `;
    // Example for optimizing candidates
    const optAcqFuncExample = `${pre}python
    <-- code goes here -->
    `;


    const QuickStart = () => (
      <div
        className="productShowcaseSection"
        id="quickstart"
        style={{textAlign: 'center'}}>
        <h2>Get Started</h2>
        <Container>
          <ol>
            <li>
              <h4>Install AEPsych:</h4>
              <a>via ##Insert here## (recommended):</a>
              <MarkdownBlock>{bash` <-- install aepsych -->`}</MarkdownBlock>
              <a>via pip:</a>
              <MarkdownBlock>{bash`<-- pip install aepsych -->`}</MarkdownBlock>
            </li>
            <li>
              <h4>Fit a model:</h4>
              <MarkdownBlock>{modelFitCodeExample}</MarkdownBlock>
            </li>
            <li>
              <h4>Construct an acquisition function:</h4>
              <MarkdownBlock>{constrAcqFuncExample}</MarkdownBlock>
            </li>
            <li>
              <h4>Optimize the acquisition function:</h4>
              <MarkdownBlock>{optAcqFuncExample}</MarkdownBlock>
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
              image: `${baseUrl}img/expanding_arrows.svg`,
              imageAlign: 'top',
              title: 'Header',
            },
            {
              content:
                'Text goes here...',
              image: `${baseUrl}img/placeholder.png`,
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

    const Reference = () => (
      <div
        className="productShowcaseSection"
        id="reference"
        style={{textAlign: 'center'}}>
        <h2>References</h2>
        <Container>
         <p>
            Lorem Ipsum is simply dummy text of the printing and typesetting industry.
            Lorem Ipsum has been the industry's standard dummy text ever since the 1500s,
            when an unknown printer took a galley of type and scrambled it to make a type
            specimen book.
          </p>
        </Container>
      </div>
    );

    return (
      <div>
        <HomeSplash siteConfig={siteConfig} language={language} />
        <div className="mainContainer">
          <Features />
          <Reference />
          <QuickStart />
        </div>
      </div>
    );
  }
}

module.exports = Index;
