# Gemma 4 Quality Evaluation

Tested: 2026-06-19T21:05:58.006Z
Backend: cpu
Variants: 4

## Overall scores

| Model | Quant | Overall | Pass rate | Load (s) |
|-------|-------|---------|-----------|----------|
| E2B-it | q4 | **0.877** | 92% | 2.8 |
| E2B-it | q4f16 | **0.876** | 92% | 68.3 |
| E4B-it | q4 | **0.897** | 96% | 15.6 |
| E4B-it | q4f16 | **0.899** | 96% | 31.7 |

## By category

### Domain writing

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 0.776 | 100% |
| E2B-it | q4f16 | 0.771 | 100% |
| E4B-it | q4 | 0.79 | 100% |
| E4B-it | q4f16 | 0.798 | 100% |

### JSON extraction

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 1 | 100% |
| E2B-it | q4f16 | 1 | 100% |
| E4B-it | q4 | 0.933 | 100% |
| E4B-it | q4f16 | 0.933 | 100% |

### MCQ (standard)

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 0.667 | 67% |
| E2B-it | q4f16 | 0.667 | 67% |
| E4B-it | q4 | 0.833 | 83% |
| E4B-it | q4f16 | 0.833 | 83% |

### Reading comp

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 1 | 100% |
| E2B-it | q4f16 | 1 | 100% |
| E4B-it | q4 | 1 | 100% |
| E4B-it | q4f16 | 1 | 100% |

### Instruction following

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 1 | 100% |
| E2B-it | q4f16 | 1 | 100% |
| E4B-it | q4 | 1 | 100% |
| E4B-it | q4f16 | 1 | 100% |

## Notes

- **24 tasks per variant** across 5 categories (9 domain writing, 3 JSON extract, 6 MCQ, 3 reading comp, 3 instruction following).
- **Languages:** English, Swedish, Turkish — topics: mortgage, legal, medical.
- **Chat template:** prompts sent as Gemma `user` turns with `return_full_text: false`.
- **Backend:** CPU (bundled onnxruntime-node via Transformers.js 4.2).
- Domain writing scored by keyword coverage, target length (~100 words), and repetition heuristics (no LLM judge).
- JSON extraction: field-level match against gold schema from long source text.
- MCQ: standard single-letter multiple choice (mortgage/legal/medical knowledge).
- Reading comp & instruction following: rule-based checks.

## Known weak spots

| Task | Issue |
|------|-------|
| `mcq-en-mortgage-ltv` | All variants answered **A** (60%) instead of **C** (80%) — arithmetic reasoning |
| `mcq-sv-legal-withdrawal` | E2B variants answered **A** instead of **B** (14 days) |

E4B models score higher overall (~0.90 vs ~0.88) with better MCQ (5/6) and slightly stronger domain writing. **q4 vs q4f16** is effectively tied on quality in this suite.

## Reproduce

```bash
npm run eval:gemma4:quality
# or single variant:
node --expose-gc scripts/eval-gemma4-quality.mjs --variant E2B-it:q4
```

Raw results: `results/eval-gemma4-quality-1781903158007.json`
