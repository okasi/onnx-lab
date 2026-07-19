import assert from 'node:assert/strict';
import test from 'node:test';
import { scoreMultimodalOutput } from '../lib/gemma4-multimodal-scoring.mjs';
import { loadMultimodalTasks } from '../lib/gemma4-multimodal-suite.mjs';

test('flat and split multimodal suite formats normalize media URLs', () => {
  const flat = loadMultimodalTasks({
    base_url: 'https://example.test/',
    tasks: [{ id: 'a', modality: 'image', media_file: 'a.png' }],
  });
  assert.equal(flat[0].media_url, 'https://example.test/a.png');

  const split = loadMultimodalTasks({
    base_url: 'https://example.test/',
    audio_tasks: [{ id: 'b', media_file: 'b.wav' }],
  }, 'audio');
  assert.equal(split[0].modality, 'audio');
  assert.equal(split[0].media_url, 'https://example.test/b.wav');
});

test('multimodal scoring reports matches and misses', () => {
  const result = scoreMultimodalOutput('A blue Volkswagen Beetle', ['blue', 'piano']);
  assert.equal(result.pass, true);
  assert.deepEqual(result.matched, ['blue']);
  assert.deepEqual(result.missing, ['piano']);
});
