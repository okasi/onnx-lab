/**
 * Score multimodal Gemma 4 outputs with simple substring checks.
 */

/**
 * @param {string} text
 * @param {string[]} expectSubstrings
 */
export function scoreMultimodalOutput(text, expectSubstrings = []) {
  const normalized = (text ?? '').toLowerCase().trim();
  if (!normalized) {
    return { pass: false, score: 0, matched: [], missing: expectSubstrings };
  }

  const matched = [];
  const missing = [];
  for (const needle of expectSubstrings) {
    if (normalized.includes(needle.toLowerCase())) {
      matched.push(needle);
    } else {
      missing.push(needle);
    }
  }

  const score = expectSubstrings.length
    ? matched.length / expectSubstrings.length
    : (normalized.length > 10 ? 1 : 0.5);

  return {
    pass: expectSubstrings.length ? matched.length >= Math.max(1, Math.ceil(expectSubstrings.length / 2)) : normalized.length > 0,
    score,
    matched,
    missing,
  };
}
