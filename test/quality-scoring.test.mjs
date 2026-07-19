import assert from 'node:assert/strict';
import test from 'node:test';
import {
  scoreClassification,
  scoreInstructionFollowing,
  scoreJsonExtraction,
  scoreMcq,
  scoreReadingComprehension,
  stripPromptEcho,
} from '../lib/gemma4-quality-scoring.mjs';

test('prompt stripping removes the complete prompt, not only its first 80 characters', () => {
  const prompt = 'A'.repeat(100);
  assert.equal(stripPromptEcho(`${prompt} completion`, prompt), 'completion');
});

test('JSON boolean scoring does not coerce the string false to true', () => {
  const task = { gold: { binding: false }, fields: ['binding'] };
  assert.equal(scoreJsonExtraction('{"binding":"false"}', task).score, 1);
  assert.equal(
    scoreJsonExtraction('{"binding":"false"}', {
      gold: { binding: true },
      fields: ['binding'],
    }).score,
    0,
  );
});

test('instruction and classification scoring enforce exact requested structure', () => {
  assert.equal(
    scoreInstructionFollowing('1. First\n2. Second', {
      type: 'numbered_count',
      count: 2,
    }).pass,
    true,
  );
  assert.equal(
    scoreInstructionFollowing('2. Second\n1. First', {
      type: 'numbered_count',
      count: 2,
    }).pass,
    false,
  );
  assert.ok(
    scoreInstructionFollowing('- a\n- b\n- c', {
      type: 'bullet_count',
      count: 2,
      prefix: '- ',
    }).score < 1,
  );
  assert.equal(scoreClassification('medical.', 'medical').pass, true);
  assert.equal(scoreClassification('medical advice', 'med').pass, false);
});

test('reading comprehension treats answer text literally', () => {
  assert.equal(scoreReadingComprehension('a+b', ['a+b']).pass, true);
  assert.equal(scoreReadingComprehension('140', ['14']).pass, false);
});

test('MCQ scoring does not treat the article A as an answer line', () => {
  assert.equal(scoreMcq('A borrower reasons through the options.\nAnswer: C', 'C').pass, true);
});
