/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the scripts directory.
 *
 * @format
 */

const React = require('react');
const fs = require('fs-extra');
const path = require('path');
const join = path.join;
const CWD = process.cwd();

const CompLibrary = require(join(
  CWD,
  '/node_modules/docusaurus/lib/core/CompLibrary.js',
));
const SideNav = require(join(
  CWD,
  '/node_modules/docusaurus/lib/core/nav/SideNav.js',
));

const Container = CompLibrary.Container;

const OVERVIEW_ID = 'demo_overview';

class DemoSidebar extends React.Component {
  render() {
    const {currentDemoID} = this.props;
    const current = {
      id: currentDemoID || OVERVIEW_ID,
    };

    const toc = [
      {
        type: 'CATEGORY',
        title: 'Demos',
        children: [
          {
            type: 'LINK',
            item: {
              permalink: 'demos/',
              id: OVERVIEW_ID,
              title: 'Overview',
            },
          },
        ],
      },
    ];

    const jsonFile = join(CWD, 'demos.json');
    const normJsonFile = path.normalize(jsonFile);
    const json = JSON.parse(fs.readFileSync(normJsonFile, {encoding: 'utf8'}));

    Object.keys(json).forEach(category => {
      const categoryItems = json[category];
      const items = [];
      categoryItems.map(item => {
        items.push({
          type: 'LINK',
          item: {
            permalink: `demos/${item.id}`,
            id: item.id,
            title: item.title,
          },
        });
      });

      toc.push({
        type: 'CATEGORY',
        title: category,
        children: items,
      });
    });

    return (
      <Container className="docsNavContainer" id="docsNav" wrapper={false}>
        <SideNav
          language={'demos'}
          root={'demos'}
          title="Demos"
          contents={toc}
          current={current}
        />
      </Container>
    );
  }
}

module.exports = DemoSidebar;
