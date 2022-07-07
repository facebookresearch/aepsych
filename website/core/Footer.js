/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */const PropTypes = require('prop-types');
const React = require('react');


function SocialFooter(props) {
  const repoUrl = `https://github.com/${props.config.organizationName}/${
    props.config.projectName
  }`;
  return (
    <div className="footerSection">
      <h5>Social</h5>
      <div className="social">
        <a
          className="github-button" // part of the https://buttons.github.io/buttons.js script in siteConfig.js
          href={repoUrl}
          data-count-href={`${repoUrl}/stargazers`}
          data-show-count="true"
          data-count-aria-label="# stargazers on GitHub"
          aria-label="Star AEPsych on GitHub">
          {props.config.projectName}
        </a>
      </div>
    </div>
  );
}

SocialFooter.propTypes = {
  config: PropTypes.object,
};

const CookieConsent = () => (
    <div className="cookie-container">
    <p>
      We use cookies to enhance your experience,
      and to analyse the use of our website. By clicking or navigating,
      you agree to allow our usage of cookies.
    </p>
    <button className="cookie-btn">
    accept
    </button>
    </div>
  )


class Footer extends React.Component {
  docUrl(doc) {
    const baseUrl = this.props.config.baseUrl;
    const docsUrl = this.props.config.docsUrl;
    const docsPart = `${docsUrl ? `${docsUrl}/` : ''}`;
    return `${baseUrl}${docsPart}${doc}`;
  }


  render() {
    return (
      <footer className="nav-footer" id="footer">
        <section className="sitemap">
          <a href={this.props.config.baseUrl} className="nav-home">
            {this.props.config.footerIcon && (
              <img
                src={this.props.config.baseUrl + this.props.config.footerIcon}
                alt={this.props.config.title}
                width="66"
                height="58"
              />
            )}
          </a>

          <div>
            <h5>Docs</h5>
            <a href={this.docUrl('introduction')}>Introduction</a>
            <a href={this.docUrl('getting_started')}>Getting Started</a>
            <a href={`${this.props.config.baseUrl}tutorials/`}>Tutorials</a>
            <a href={`${this.props.config.baseUrl}api/`}>API Reference</a>
          </div>

          <div>
            <h5>Legal</h5>
            <a
              href="https://opensource.facebook.com/legal/privacy/"
              target="_blank"
              rel="noreferrer noopener">
              Privacy
            </a>
            <a
              href="https://opensource.facebook.com/legal/terms/"
              target="_blank"
              rel="noreferrer noopener">
              Terms
            </a>
          </div>

          <div>
            <SocialFooter config={this.props.config} />

          </div>

        </section>

        <a
          href="https://opensource.facebook.com/"
          target="_blank"
          rel="noreferrer noopener"
          className="fbOpenSource">
          <img
            src={`${this.props.config.baseUrl}img/oss_logo.png`}
            alt="Facebook Open Source"
            width="170"
            height="45"
          />
        </a>
        <section className="copyright">{this.props.config.copyright}  Built with Docusaurus.</section>
        <CookieConsent />
      </footer>
    );
  }
}

module.exports = Footer;
