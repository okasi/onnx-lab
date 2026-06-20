# Gemma 4 — ORT 1.28 vs bundled 1.24 (CPU speed)

Compared **onnxruntime-node 1.28.0** (local build) against **bundled 1.24.3** on CPU for the four production quants.

**Setup:** 6 prompts × 8 new tokens, greedy, `device: cpu`, models `onnx-community/gemma-4-{E2B,E4B}-it-ONNX`.

**Runs:** 3 independent benchmark passes per ORT version (same VM, same day).

## 3-run averages (mean ± std, min–max)

| Model | Quant | ORT | ms/prompt | tok/s | RSS (MB) |
|-------|-------|-----|-----------|-------|----------|
| E2B-it | q4 | 1.24 | 944.5 ± 43.7 (886–990) | 8.49 ± 0.40 (8.08–9.03) | 1,933 |
| E2B-it | q4 | **1.28** | **881.0 ± 47.1 (819–934)** | **9.11 ± 0.49 (8.57–9.76)** | 1,930 |
| E2B-it | q4f16 | 1.24 | 910.6 ± 19.6 (897–938) | 8.79 ± 0.18 (8.53–8.92) | 2,337 |
| E2B-it | q4f16 | 1.28 | 933.2 ± 71.2 (881–1034) | 8.62 ± 0.62 (7.74–9.08) | 2,339 |
| E4B-it | q4 | 1.24 | 1812.1 ± 97.3 (1729–1949) | 4.43 ± 0.23 (4.11–4.63) | 3,376 |
| E4B-it | q4 | 1.28 | 1844.5 ± 127.2 (1676–1983) | 4.36 ± 0.31 (4.03–4.77) | 3,363 |
| E4B-it | q4f16 | 1.24 | 1830.4 ± 120.7 (1736–2001) | 4.39 ± 0.28 (4.00–4.61) | 3,999 |
| E4B-it | q4f16 | 1.28 | 1833.3 ± 116.2 (1735–1997) | 4.38 ± 0.26 (4.01–4.61) | 3,977 |

## Average delta (1.28 vs 1.24, 3 runs)

| Variant | tok/s change | ms/prompt change |
|---------|--------------|------------------|
| **E2B q4** | **+7.3%** | −63 ms (−6.7%) |
| E2B q4f16 | −1.9% | +23 ms |
| E4B q4 | −1.7% | +32 ms |
| E4B q4f16 | −0.2% | +3 ms |

## Per-run detail — E2B q4 (the only consistent winner)

| Run | ORT 1.24 tok/s | ORT 1.28 tok/s | 1.28 faster? |
|-----|----------------|----------------|--------------|
| 1 | 8.08 | 8.99 | yes (+11%) |
| 2 | 8.57 | 8.57 | tie |
| 3 | 9.03 | 9.76 | yes (+8%) |

All three 1.28 E2B-q4 runs are at or above the 1.24 mean (8.49). 1.28 mean ms/prompt (881) is below every 1.24 run (886–990).

## Takeaways

- **E2B q4** is the only variant with a repeatable win: ~7% higher throughput across 3 runs, not a one-off.
- **E4B** (both quants) and **E2B q4f16** are within noise — overlapping ranges, &lt;2% average difference.
- **Peak RSS** unchanged on all variants.
- Run-to-run variance is ~±5% tok/s for E2B and ~±6% for E4B; use 3+ runs when comparing small deltas.

## Raw results

| ORT | Run | File |
|-----|-----|------|
| 1.24 | 1 | `results/benchmark-gemma4-speed-4variants.json` |
| 1.24 | 2 | `results/benchmark-gemma4-speed-ort124-run2.json` |
| 1.24 | 3 | `results/benchmark-gemma4-speed-ort124-run3.json` |
| 1.28 | 1 | `results/benchmark-gemma4-speed-ort128.json` |
| 1.28 | 2 | `results/benchmark-gemma4-speed-ort128-run2.json` |
| 1.28 | 3 | `results/benchmark-gemma4-speed-ort128-run3.json` |

Reproduce:

```bash
npm run build:ort
# add package.json overrides to file:vendor/onnxruntime/js/{node,common,web}
ONNXRUNTIME_NODE_INSTALL=skip npm install
node --expose-gc scripts/benchmark-gemma4.mjs --model E2B-it,E4B-it --dtype q4,q4f16 --backend cpu
```
