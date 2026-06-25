/**
 * Heuristic scorers for Gemma 4 quality evaluation (no external judge model).
 */

export function round(n, digits = 3) {
  return Number(Number(n).toFixed(digits));
}

export function stripPromptEcho(text, prompt) {
  if (!text) return '';
  let out = text.trim();
  const prefix = prompt.trim().slice(0, 80);
  if (out.startsWith(prefix)) {
    out = out.slice(prefix.length).trim();
  }
  return out;
}

export function wordCount(text) {
  return (text.match(/\S+/g) ?? []).length;
}

export function uniqueWordRatio(text) {
  const words = (text.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? []);
  if (!words.length) return 0;
  return new Set(words).size / words.length;
}

export function keywordHitRate(text, keywords) {
  if (!keywords?.length) return 0;
  const lower = text.toLowerCase();
  const hits = keywords.filter((k) => lower.includes(k.toLowerCase()));
  return hits.length / keywords.length;
}

export function repetitionPenalty(text) {
  const words = (text.toLowerCase().match(/[\p{L}\p{N}']+/gu) ?? []);
  if (words.length < 8) return 0;
  const trigrams = new Map();
  let repeats = 0;
  for (let i = 0; i < words.length - 2; i += 1) {
    const tri = words.slice(i, i + 3).join(' ');
    trigrams.set(tri, (trigrams.get(tri) ?? 0) + 1);
    if (trigrams.get(tri) === 2) repeats += 1;
  }
  return Math.min(1, repeats / Math.max(1, words.length / 12));
}

export function scoreDomainWriting(text, task, config) {
  const body = stripPromptEcho(text, task.prompt);
  const words = wordCount(body);
  const minW = config.min_words ?? 60;
  const targetW = config.target_words ?? 100;
  const maxW = config.max_words ?? 220;

  let lengthScore = 0;
  if (words >= minW && words <= maxW) {
    lengthScore = 1 - Math.min(1, Math.abs(words - targetW) / targetW);
  } else if (words > 20) {
    lengthScore = 0.25;
  }

  const kw = keywordHitRate(body, task.keywords);
  const uniq = uniqueWordRatio(body);
  const rep = repetitionPenalty(body);
  const coherence = Math.max(0, uniq - rep);

  const score = round(
    0.35 * lengthScore + 0.35 * kw + 0.3 * Math.min(1, coherence),
  );

  return {
    score,
    words,
    keyword_hit_rate: round(kw),
    unique_word_ratio: round(uniq),
    repetition_penalty: round(rep),
    pass: score >= 0.55 && words >= minW * 0.7,
  };
}

export function extractJsonObject(text) {
  if (!text) return { ok: false, error: 'empty response' };
  let slice = text.trim();
  const fence = slice.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fence) slice = fence[1].trim();
  const start = slice.indexOf('{');
  const end = slice.lastIndexOf('}');
  if (start === -1 || end <= start) {
    return { ok: false, error: 'no JSON object found' };
  }
  try {
    const parsed = JSON.parse(slice.slice(start, end + 1));
    return { ok: true, value: parsed };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e) };
  }
}

function normalizeString(s) {
  return String(s)
    .toLowerCase()
    .normalize('NFD')
    .replace(/\p{M}/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function fieldMatch(got, expected) {
  if (expected == null) return got == null;
  if (typeof expected === 'number') {
    const n = Number(got);
    return Number.isFinite(n) && Math.abs(n - expected) < 0.01;
  }
  if (typeof expected === 'boolean') {
    return Boolean(got) === expected;
  }
  const g = normalizeString(got);
  const e = normalizeString(expected);
  if (g === e) return true;
  if (g.includes(e) || e.includes(g)) return true;
  return false;
}

export function scoreJsonExtraction(text, task) {
  const { ok, value, error } = extractJsonObject(text);
  if (!ok) {
    return { score: 0, parse_ok: false, field_accuracy: 0, error, parsed: null };
  }

  const fields = task.fields ?? Object.keys(task.gold);
  let correct = 0;
  const perField = {};
  for (const field of fields) {
    const match = fieldMatch(value[field], task.gold[field]);
    perField[field] = { expected: task.gold[field], got: value[field], match };
    if (match) correct += 1;
  }

  const fieldAccuracy = fields.length ? correct / fields.length : 0;
  const score = round(fieldAccuracy);

  return {
    score,
    parse_ok: true,
    field_accuracy: round(fieldAccuracy),
    fields_correct: correct,
    fields_total: fields.length,
    per_field: perField,
    pass: fieldAccuracy >= 0.8,
  };
}

export function scoreMcq(text, correct) {
  const body = (text ?? '').trim();
  // Prefer explicit answer lines; fall back to last standalone letter.
  const answerLine = body.match(/(?:^|\n)\s*(?:answer|svar|cevap)?\s*:?\s*([A-Da-d])\b/i);
  const letters = [...body.matchAll(/\b([A-Da-d])\b/g)].map((m) => m[1].toUpperCase());
  const letter = answerLine?.[1]?.toUpperCase() ?? letters.at(-1) ?? null;
  const ok = letter === correct.toUpperCase();
  return {
    score: ok ? 1 : 0,
    predicted: letter,
    correct: correct.toUpperCase(),
    pass: ok,
  };
}

export function scoreReadingComprehension(text, goldAnswers) {
  const body = normalizeString(stripPromptEcho(text, ''));
  const ok = goldAnswers.some((a) => {
    const g = normalizeString(a);
    return body === g || body.includes(g) || new RegExp(`\\b${g}\\b`).test(body);
  });
  return {
    score: ok ? 1 : 0,
    response: text?.trim().slice(0, 80),
    pass: ok,
  };
}

function splitSentences(text) {
  return text
    .split(/(?<=[.!?…])\s+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 5);
}

export function scoreInstructionFollowing(text, rules) {
  const body = stripPromptEcho(text, '');
  if (!rules?.type) return { score: 0, pass: false, detail: 'no rules' };

  switch (rules.type) {
    case 'bullet_count': {
      const lines = body.split('\n').map((l) => l.trim()).filter(Boolean);
      const bullets = lines.filter((l) => l.startsWith(rules.prefix ?? '- '));
      const ok = bullets.length === rules.count;
      return {
        score: ok ? 1 : round(bullets.length / rules.count, 2),
        bullet_count: bullets.length,
        expected: rules.count,
        pass: ok,
      };
    }
    case 'sentence_count': {
      const sentences = splitSentences(body);
      const ok = sentences.length === rules.count;
      return {
        score: ok ? 1 : round(Math.min(1, sentences.length / rules.count), 2),
        sentence_count: sentences.length,
        expected: rules.count,
        pass: ok,
      };
    }
    case 'exact_one_of': {
      const trimmed = body.trim();
      const ok = rules.options.some((o) => trimmed === o || trimmed.toLowerCase() === o.toLowerCase());
      return { score: ok ? 1 : 0, response: trimmed.slice(0, 40), pass: ok };
    }
    case 'numbered_count': {
      const lines = body.split('\n').map((l) => l.trim()).filter(Boolean);
      const numbered = lines.filter((l, i) => l.match(new RegExp(`^${i + 1}[.)\\s]`)) || l.startsWith(`${i + 1}.`));
      const alt = lines.filter((l) => /^\d+[.)]\s/.test(l));
      const count = Math.max(numbered.length, alt.length);
      const ok = count === rules.count;
      return { score: ok ? 1 : round(Math.min(1, count / rules.count), 2), numbered_count: count, expected: rules.count, pass: ok };
    }
    default: {
      const _exhaustive = rules.type;
      return { score: 0, pass: false, detail: `unknown rule ${_exhaustive}` };
    }
  }
}

export function scoreSummarization(text, task) {
  const body = stripPromptEcho(text, '');
  const sentences = splitSentences(body);
  const target = task.sentence_count ?? 2;
  const sentenceOk = sentences.length === target;
  const kw = keywordHitRate(body, task.required_keywords ?? []);
  const score = round((sentenceOk ? 0.5 : 0.15) + 0.5 * kw);
  return {
    score,
    sentence_count: sentences.length,
    expected_sentences: target,
    keyword_hit_rate: round(kw),
    pass: sentenceOk && kw >= 0.33,
  };
}

export function scoreClassification(text, expected) {
  const body = normalizeString(text).split(/\s+/)[0] ?? '';
  const ok = body === normalizeString(expected) || body.includes(normalizeString(expected));
  return {
    score: ok ? 1 : 0,
    predicted: text?.trim().slice(0, 30),
    expected,
    pass: ok,
  };
}

export function aggregateCategory(results) {
  if (!results.length) return { mean_score: 0, pass_rate: 0, count: 0 };
  const mean = results.reduce((s, r) => s + r.score, 0) / results.length;
  const passRate = results.filter((r) => r.pass).length / results.length;
  return {
    mean_score: round(mean),
    pass_rate: round(passRate),
    count: results.length,
  };
}

export function overallWeightedScore(categorySummaries, weights) {
  let total = 0;
  let wSum = 0;
  for (const [cat, summary] of Object.entries(categorySummaries)) {
    const w = weights[cat] ?? 0;
    total += w * summary.mean_score;
    wSum += w;
  }
  return wSum > 0 ? round(total / wSum) : 0;
}
