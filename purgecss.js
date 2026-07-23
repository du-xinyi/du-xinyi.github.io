import { promises as fs } from 'fs';
import { PurgeCSS } from 'purgecss';

const DIST_PATH = '_sass/vendors';
const output = `${DIST_PATH}/_bootstrap.scss`;
const requiredSelectors = [
  '.dropdown-menu',
  '.dropdown-item',
  '.dropup',
  '.btn-group'
];

const config = {
  content: ['_includes/**/*.html', '_layouts/**/*.html', '_javascript/**/*.js'],
  css: ['node_modules/bootstrap/dist/css/bootstrap.min.css'],
  keyframes: true,
  variables: true,
  // The `safelist` should be changed appropriately for future development
  safelist: {
    standard: [
      /^collaps/,
      /^w-/,
      'shadow',
      'border',
      'kbd',
      /^dropdown/,
      /^dropup$/,
      /^btn-group/
    ],
    greedy: [/^col-/, /tooltip/]
  }
};

function main() {
  fs.rm(DIST_PATH, { recursive: true, force: true })
    .then(() => fs.mkdir(DIST_PATH))
    .then(() => new PurgeCSS().purge(config))
    .then((result) => {
      const css = result[0].css;
      const missing = requiredSelectors.filter(
        (selector) => !css.includes(selector)
      );

      if (missing.length > 0) {
        throw new Error(`Missing required selectors: ${missing.join(', ')}`);
      }

      return fs.writeFile(output, css);
    })
    .catch((err) => {
      console.error('Error during PurgeCSS process:', err);
    });
}

main();
