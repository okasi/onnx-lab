import assert from 'node:assert/strict';
import test from 'node:test';
import { projectPath, readJson } from '../lib/benchmark-support.mjs';
import { buildCorpus } from '../scripts/generate-corpus.mjs';

test('corpus generator exactly reproduces the committed corpus', async () => {
  const committed = await readJson(projectPath('data', 'benchmark-corpus.json'));
  assert.deepEqual(buildCorpus(), committed);
});
