export function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) {
    return 0;
  }
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i += 1) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 0 : dot / denom;
}

export function mean(values) {
  if (values.length === 0) {
    return 0;
  }
  return values.reduce((a, b) => a + b, 0) / values.length;
}

function round(n, digits = 4) {
  return Number(n.toFixed(digits));
}

function reciprocalRank(ranked, predicate) {
  const idx = ranked.findIndex(predicate);
  return idx === -1 ? 0 : 1 / (idx + 1);
}

function recallAtK(ranked, k, predicate) {
  return ranked.slice(0, k).some(predicate) ? 1 : 0;
}

/**
 * Retrieval evaluation with multiple relevance definitions.
 * - topic_any: same topic, any language
 * - topic_cross_lang: same topic, different language (SV↔TR)
 * - topic_same_lang: same topic, same language (excluding self)
 */
function evaluateRetrieval(documents, embeddings) {
  const tasks = {
    topic_any: (q, d) => d.topic === q.topic && d.id !== q.id,
    topic_cross_lang: (q, d) => d.topic === q.topic && d.language !== q.language,
    topic_same_lang: (q, d) =>
      d.topic === q.topic && d.language === q.language && d.id !== q.id,
  };

  const accumulators = Object.fromEntries(
    Object.keys(tasks).map((name) => [
      name,
      { r1: 0, r3: 0, r5: 0, r10: 0, mrr10: 0, count: 0, byTopic: {} },
    ]),
  );

  for (const query of documents) {
    const queryVector = embeddings.get(query.id);
    if (!queryVector) {
      continue;
    }
    const ranked = documents
      .filter((document) => document.id !== query.id && embeddings.has(document.id))
      .map((document) => ({
        id: document.id,
        topic: document.topic,
        language: document.language,
        sim: cosineSimilarity(queryVector, embeddings.get(document.id)),
      }))
      .sort((a, b) => b.sim - a.sim);

    for (const [taskName, isRelevant] of Object.entries(tasks)) {
      const relevant = (item) => isRelevant(query, item);
      if (!ranked.some(relevant)) {
        continue;
      }
      const accumulator = accumulators[taskName];
      const metrics = {
        r1: recallAtK(ranked, 1, relevant),
        r3: recallAtK(ranked, 3, relevant),
        r5: recallAtK(ranked, 5, relevant),
        r10: recallAtK(ranked, 10, relevant),
        mrr10: reciprocalRank(ranked.slice(0, 10), relevant),
      };
      accumulator.count += 1;
      for (const [name, value] of Object.entries(metrics)) {
        accumulator[name] += value;
      }
      accumulator.byTopic[query.topic] ??= {
        r1: 0,
        r3: 0,
        r5: 0,
        mrr10: 0,
        count: 0,
      };
      const topic = accumulator.byTopic[query.topic];
      topic.count += 1;
      topic.r1 += metrics.r1;
      topic.r3 += metrics.r3;
      topic.r5 += metrics.r5;
      topic.mrr10 += metrics.mrr10;
    }
  }

  return Object.fromEntries(
    Object.entries(accumulators).map(([taskName, accumulator]) => {
      const denominator = accumulator.count || 1;
      return [
        taskName,
        {
          queries_evaluated: accumulator.count,
          recall_at_1: round(accumulator.r1 / denominator),
          recall_at_3: round(accumulator.r3 / denominator),
          recall_at_5: round(accumulator.r5 / denominator),
          recall_at_10: round(accumulator.r10 / denominator),
          mrr_at_10: round(accumulator.mrr10 / denominator),
          by_topic: Object.fromEntries(
            Object.entries(accumulator.byTopic).map(([topic, values]) => [
              topic,
              {
                recall_at_1: round(values.r1 / values.count),
                recall_at_3: round(values.r3 / values.count),
                recall_at_5: round(values.r5 / values.count),
                mrr_at_10: round(values.mrr10 / values.count),
                queries: values.count,
              },
            ]),
          ),
        },
      ];
    }),
  );
}

export function computeQuality(embeddings, documents, queryPairs) {
  const sameTopicSims = [];
  const diffTopicSims = [];
  const crossLangByTopic = Object.fromEntries(
    [...new Set(documents.map((document) => document.topic))].map((topic) => [topic, []]),
  );

  for (let i = 0; i < documents.length; i += 1) {
    for (let j = i + 1; j < documents.length; j += 1) {
      const a = documents[i];
      const b = documents[j];
      const va = embeddings.get(a.id);
      const vb = embeddings.get(b.id);
      if (!va || !vb) {
        continue;
      }
      const sim = cosineSimilarity(va, vb);
      if (a.topic === b.topic) {
        sameTopicSims.push(sim);
        if (a.language !== b.language) {
          crossLangByTopic[a.topic].push(sim);
        }
      } else {
        diffTopicSims.push(sim);
      }
    }
  }

  const pairScores = queryPairs
    .map((pair) => {
      const sv = embeddings.get(pair.sv_doc_id);
      const tr = embeddings.get(pair.tr_doc_id);
      if (!sv || !tr) {
        return null;
      }
      return {
        pair_id: pair.id,
        topic: pair.topic,
        cosine_similarity: cosineSimilarity(sv, tr),
      };
    })
    .filter(Boolean);

  const pairValues = pairScores.map((p) => p.cosine_similarity);
  const cohesion = mean(sameTopicSims);
  const separation = mean(diffTopicSims);
  const retrieval = evaluateRetrieval(documents, embeddings);

  const crossLangR5 = retrieval.topic_cross_lang.recall_at_5;
  const anyTopicR5 = retrieval.topic_any.recall_at_5;

  return {
    topic_cohesion_mean: round(cohesion),
    topic_separation_mean: round(separation),
    topic_discrimination: round(cohesion - separation),
    cross_lingual_pairs: {
      count: pairScores.length,
      mean_cosine: round(mean(pairValues)),
      min_cosine: pairScores.length ? round(Math.min(...pairValues)) : 0,
      max_cosine: pairScores.length ? round(Math.max(...pairValues)) : 0,
      by_topic: Object.fromEntries(
        Object.entries(crossLangByTopic).map(([topic, vals]) => [
          topic,
          { count: vals.length, mean_cosine: round(mean(vals)) },
        ]),
      ),
      pairs: pairScores.map((pair) => ({
        ...pair,
        cosine_similarity: round(pair.cosine_similarity, 4),
      })),
    },
    retrieval,
    // Legacy field kept for compatibility with summary tables
    recall_at_5: anyTopicR5,
    cross_lingual_recall_at_5: crossLangR5,
    composite_score: round(
      mean(pairValues) * 0.35 +
        (cohesion - separation) * 0.25 +
        crossLangR5 * 0.25 +
        anyTopicR5 * 0.15,
    ),
  };
}
