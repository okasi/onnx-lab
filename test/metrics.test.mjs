import assert from 'node:assert/strict';
import test from 'node:test';
import { computeQuality, cosineSimilarity } from '../lib/metrics.mjs';

test('cosine similarity handles invalid and zero vectors', () => {
  assert.equal(cosineSimilarity([1], [1, 2]), 0);
  assert.equal(cosineSimilarity([0, 0], [1, 0]), 0);
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1);
});

test('retrieval metrics divide by queries that have a relevant document', () => {
  const documents = [
    { id: 'sv-x', topic: 'x', language: 'sv' },
    { id: 'tr-x', topic: 'x', language: 'tr' },
    { id: 'sv-y', topic: 'y', language: 'sv' },
  ];
  const embeddings = new Map([
    ['sv-x', [1, 0]],
    ['tr-x', [0.9, 0.1]],
    ['sv-y', [0, 1]],
  ]);
  const quality = computeQuality(embeddings, documents, [
    { id: 'pair-x', topic: 'x', sv_doc_id: 'sv-x', tr_doc_id: 'tr-x' },
  ]);
  assert.equal(quality.retrieval.topic_any.queries_evaluated, 2);
  assert.equal(quality.retrieval.topic_any.recall_at_1, 1);
  assert.equal(quality.retrieval.topic_same_lang.queries_evaluated, 0);
});

test('missing embeddings are ignored without crashing', () => {
  const documents = [
    { id: 'a', topic: 'x', language: 'sv' },
    { id: 'b', topic: 'x', language: 'tr' },
  ];
  const quality = computeQuality(new Map([['a', [1, 0]]]), documents, []);
  assert.equal(quality.cross_lingual_pairs.count, 0);
  assert.equal(quality.retrieval.topic_any.queries_evaluated, 0);
});
