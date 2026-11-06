// Lint content files for formatting problems that the build won't catch.
//
// Recipes (_recipes/**/*.md):
//   - front matter has a title, and the body has a "制作过程" section
//   - ingredient tables are well formed and column-aligned
//   - numbered steps (and P.S. notes) are numbered 1, 2, 3, ...
//   - a line followed by a numbered line ends with two spaces, otherwise
//     Markdown joins the two lines into one
// Posts (_posts/*.md):
//   - posts with `math: true` have balanced `$$` block delimiters
//
// Usage: node tools/content-lint.mjs

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const errors = [];

function report(file, line, message) {
  errors.push(`${relative(root, file)}:${line}: ${message}`);
}

function walk(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return walk(path);
    return name.endsWith('.md') ? [path] : [];
  });
}

// Display width: East Asian wide/fullwidth characters count as 2 columns.
const WIDE = [
  [0x1100, 0x115f], [0x2e80, 0x303e], [0x3041, 0x33ff], [0x3400, 0x4dbf],
  [0x4e00, 0x9fff], [0xa000, 0xa4cf], [0xac00, 0xd7a3], [0xf900, 0xfaff],
  [0xfe30, 0xfe4f], [0xff00, 0xff60], [0xffe0, 0xffe6], [0x1f300, 0x1f64f],
  [0x1f900, 0x1f9ff], [0x20000, 0x3fffd]
];

function width(text) {
  let total = 0;
  for (const char of text) {
    const cp = char.codePointAt(0);
    total += WIDE.some(([lo, hi]) => cp >= lo && cp <= hi) ? 2 : 1;
  }
  return total;
}

function cells(line) {
  return line.trim().replace(/^\||\|$/g, '').split('|').map((cell) => cell.trim());
}

// Render a single-header ingredient table the way the existing recipes are
// laid out: left-aligned names, right-aligned quantities, minimum width 4.
function renderTable(head, rows) {
  const c1 = Math.max(width(head), ...rows.map(([name]) => width(name)));
  const c2 = Math.max(4, ...rows.map(([, qty]) => width(qty)));
  const pad = (s, n) => s + ' '.repeat(n - width(s));
  const padLeft = (s, n) => ' '.repeat(n - width(s)) + s;
  return [
    `| ${pad(head, c1)} |`,
    `| :${'-'.repeat(c1 - 1)} | ${'-'.repeat(c2 - 1)}: |`,
    ...rows.map(([name, qty]) => `| ${pad(name, c1)} | ${padLeft(qty, c2)} |`)
  ];
}

function lintTables(file, lines) {
  for (let i = 0; i < lines.length; i++) {
    if (!lines[i].startsWith('|') || (i > 0 && lines[i - 1].startsWith('|'))) continue;

    let end = i;
    while (end < lines.length && lines[end].startsWith('|')) end++;
    const block = lines.slice(i, end);
    const header = cells(block[0]);

    if (block.length < 3 || !/^\|\s*:?-+:?\s*\|/.test(block[1])) {
      report(file, i + 1, 'table needs a header, a separator row and at least one row');
    } else if (header.length === 1) {
      const rows = block.slice(2).map(cells);
      const bad = rows.findIndex((row) => row.length !== 2);
      if (bad !== -1) {
        report(file, i + 3 + bad, 'ingredient row must have exactly 2 columns');
      } else {
        const expected = renderTable(header[0], rows);
        const diff = expected.findIndex((line, k) => line !== block[k]);
        if (diff !== -1) {
          report(file, i + 1 + diff, `table is not aligned, expected: ${expected[diff]}`);
        }
      }
    } else {
      block.forEach((line, k) => {
        if (k !== 1 && cells(line).length !== header.length) {
          report(file, i + 1 + k, `row must have ${header.length} columns`);
        }
      });
    }
    i = end - 1;
  }
}

function lintRecipe(file) {
  const text = readFileSync(file, 'utf8');
  const lines = text.split('\n');
  const frontMatter = text.match(/^---\n([\s\S]*?)\n---\n/);

  if (!frontMatter || !/^title:\s*\S/m.test(frontMatter[1])) {
    report(file, 1, 'front matter must define a title');
  }
  if (!lines.some((line) => line.startsWith('制作过程'))) {
    report(file, 1, 'missing "制作过程" section');
  }

  lintTables(file, lines);

  let expected = 0;
  lines.forEach((line, i) => {
    if (line.startsWith('制作过程') || line.startsWith('P.S')) {
      expected = 1;
      return;
    }
    const step = line.match(/^(\d+)、/);
    if (step && expected) {
      if (Number(step[1]) !== expected) {
        report(file, i + 1, `step should be numbered ${expected}, found ${step[1]}`);
      }
      expected = Number(step[1]) + 1;
    }
    const next = lines[i + 1];
    if (line.trim() && next !== undefined && /^\d+、/.test(next) && !line.endsWith('  ')) {
      report(file, i + 1, 'line before a numbered step must end with two spaces');
    }
  });
}

function lintPost(file) {
  const text = readFileSync(file, 'utf8');
  if (!/^math:\s*true\s*$/im.test(text)) return;

  const blocks = text.split('\n').filter((line) => line.trim() === '$$').length;
  if (blocks % 2 !== 0) report(file, 1, 'unbalanced $$ math block delimiters');
}

walk(join(root, '_recipes')).forEach(lintRecipe);
walk(join(root, '_posts')).forEach(lintPost);

if (errors.length) {
  console.error(errors.join('\n'));
  console.error(`\n${errors.length} content problem(s) found.`);
  process.exit(1);
}
console.log('Content lint passed.');
