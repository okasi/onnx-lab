export function cosineSimilarity(a, b) {
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
function evaluateRetrieval(queries, corpus, embeddings) {
  const tasks = {
    topic_any: (q, d) => d.topic === q.topic && d.id !== q.id,
    topic_cross_lang: (q, d) => d.topic === q.topic && d.language !== q.language,
    topic_same_lang: (q, d) =>
      d.topic === q.topic && d.language === q.language && d.id !== q.id,
  };

  const results = {};

  for (const [taskName, isRelevant] of Object.entries(tasks)) {
    let r1 = 0;
    let r3 = 0;
    let r5 = 0;
    let r10 = 0;
    let mrr10 = 0;
    let queriesWithRelevant = 0;
    const byTopic = {};

    for (const query of queries) {
      const qv = embeddings.get(query.id);
      if (!qv) {
        continue;
      }

      const ranked = corpus
        .filter((d) => d.id !== query.id)
        .map((d) => ({
          id: d.id,
          topic: d.topic,
          language: d.language,
          sim: cosineSimilarity(qv, embeddings.get(d.id)),
        }))
        .sort((a, b) => b.sim - a.sim);

      const hasRelevant = ranked.some((r) => isRelevant(query, r));
      if (!hasRelevant) {
        continue;
      }

      queriesWithRelevant += 1;
      const pred = (r) => isRelevant(query, r);

      r1 += recallAtK(ranked, 1, pred);
      r3 += recallAtK(ranked, 3, pred);
      r5 += recallAtK(ranked, 5, pred);
      r10 += recallAtK(ranked, 10, pred);
      mrr10 += reciprocalRank(ranked.slice(0, 10), pred);

      byTopic[query.topic] ??= { r1: 0, r3: 0, r5: 0, mrr10: 0, count: 0 };
      byTopic[query.topic].count += 1;
      byTopic[query.topic].r1 += recallAtK(ranked, 1, pred);
      byTopic[query.topic].r3 += recallAtK(ranked, 3, pred);
      byTopic[query.topic].r5 += recallAtK(ranked, 5, pred);
      byTopic[query.topic].mrr10 += reciprocalRank(ranked.slice(0, 10), pred);
    }

    const denom = queries.length;
    results[taskName] = {
      queries_evaluated: queriesWithRelevant,
      recall_at_1: round(r1 / denom),
      recall_at_3: round(r3 / denom),
      recall_at_5: round(r5 / denom),
      recall_at_10: round(r10 / denom),
      mrr_at_10: round(mrr10 / denom),
      by_topic: Object.fromEntries(
        Object.entries(byTopic).map(([topic, v]) => [
          topic,
          {
            recall_at_1: round(v.r1 / v.count),
            recall_at_3: round(v.r3 / v.count),
            recall_at_5: round(v.r5 / v.count),
            mrr_at_10: round(v.mrr10 / v.count),
            queries: v.count,
          },
        ]),
      ),
    };
  }

  return results;
}

export function computeQuality(embeddings, documents, queryPairs) {
  const sameTopicSims = [];
  const diffTopicSims = [];
  const crossLangByTopic = { mortgage: [], legal: [], medical: [] };

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
        cosine_similarity: round(cosineSimilarity(sv, tr), 4),
      };
    })
    .filter(Boolean);

  const pairValues = pairScores.map((p) => p.cosine_similarity);
  const cohesion = mean(sameTopicSims);
  const separation = mean(diffTopicSims);
  const retrieval = evaluateRetrieval(documents, documents, embeddings);

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
      pairs: pairScores,
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
