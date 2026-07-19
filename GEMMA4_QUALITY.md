# Gemma 4 Quality Evaluation

Tested: 2026-06-19T22:48:07.387Z
Backend: cpu
Variants: 4

## Overall scores

| Model | Quant | Overall | Pass rate | Load (s) |
|-------|-------|---------|-----------|----------|
| E2B-it | q4 | **0.873** | 92% | 4.0 |
| E2B-it | q4f16 | **0.873** | 92% | 4.2 |
| E4B-it | q4 | **0.91** | 96% | 5.6 |
| E4B-it | q4f16 | **0.911** | 96% | 7.0 |

## By category

### Domain writing

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 0.784 | 100% |
| E2B-it | q4f16 | 0.783 | 100% |
| E4B-it | q4 | 0.809 | 100% |
| E4B-it | q4f16 | 0.814 | 100% |

### JSON extraction

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 0.956 | 100% |
| E2B-it | q4f16 | 0.956 | 100% |
| E4B-it | q4 | 0.911 | 89% |
| E4B-it | q4f16 | 0.911 | 89% |

### MCQ (standard)

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 0.778 | 78% |
| E2B-it | q4f16 | 0.778 | 78% |
| E4B-it | q4 | 0.889 | 89% |
| E4B-it | q4f16 | 0.889 | 89% |

### Reading comp

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 0.889 | 89% |
| E2B-it | q4f16 | 0.889 | 89% |
| E4B-it | q4 | 1 | 100% |
| E4B-it | q4f16 | 1 | 100% |

### Instruction following

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 1 | 100% |
| E2B-it | q4f16 | 1 | 100% |
| E4B-it | q4 | 1 | 100% |
| E4B-it | q4f16 | 1 | 100% |

### Summarization

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 0.741 | 89% |
| E2B-it | q4f16 | 0.741 | 89% |
| E4B-it | q4 | 0.815 | 100% |
| E4B-it | q4f16 | 0.815 | 100% |

### Classification

| Model | Quant | Mean score | Pass rate |
|-------|-------|------------|-----------|
| E2B-it | q4 | 1 | 100% |
| E2B-it | q4f16 | 1 | 100% |
| E4B-it | q4 | 1 | 100% |
| E4B-it | q4f16 | 1 | 100% |

## Notes

- **78 tasks per variant** (v2 suite): 18 writing, 9 JSON, 18 MCQ, 9 reading, 9 instruction, 9 summarization, 6 classification.
- **Languages:** English, Swedish, Turkish — topics: mortgage, legal, medical.
- **Chat template** with greedy decode on CPU (Transformers.js 4.2 + pinned ORT 1.27).
- Suite source: `data/gemma4-quality-suite.json`

## Known weak spots (v2)

| Task | Issue |
|------|-------|
| `mcq-en-mortgage-ltv` | All variants fail LTV arithmetic (80% vs wrong answer) |
| `mcq-en-mortgage-payment` | Borrower age vs principal schedule reasoning |
| `mcq-sv-legal-withdrawal` | E2B misses 14-day ångerfrist |
| `rc-en-mortgage` | E2B fails $320,000 loan amount extraction |
| `json-en-legal` | E4B sometimes mis-parses `binding` boolean |
| `sum-tr-legal` | E2B misses keyword coverage in 2-sentence Turkish legal summary |

**E4B > E2B** on overall (~0.91 vs ~0.87). **q4 ≈ q4f16** on quality.

Raw results: `results/eval-gemma4-quality-1781909287392.json`
