const CWD = process.cwd();

const React = require('react');
const Demo = require(`${CWD}/core/Demo.js`);

class DemoPage extends React.Component {
  render() {
      const {config: siteConfig} = this.props;
      const {baseUrl} = siteConfig;
      return <Demo baseUrl={baseUrl} demoID="ThrowOptimizerDemo" hasWinDemo="True"
        hasMacDemo="False"/>;
  }
}

module.exports = DemoPage;
