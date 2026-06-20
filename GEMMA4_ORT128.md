# Gemma 4 — ORT 1.28 vs bundled 1.24 (CPU speed)

Compared **onnxruntime-node 1.28.0** (local build from `vendor/onnxruntime`, `main` @ 1.28.0) against the **bundled 1.24.3** default in `@huggingface/transformers`.

**Setup:** same 6 prompts × 8 new tokens, greedy, `device: cpu`, models `onnx-community/gemma-4-{E2B,E4B}-it-ONNX` with `q4` / `q4f16`.

| Model | Quant | ORT | ms/prompt | tok/s | RSS (MB) |
|-------|-------|-----|-----------|-------|----------|
| E2B-it | q4 | 1.24 | 990 | 8.08 | 1,955 |
| E2B-it | q4 | **1.28** | **890** | **8.99** | 1,959 |
| E2B-it | q4f16 | 1.24 | 938 | 8.53 | 2,342 |
| E2B-it | q4f16 | 1.28 | 1,034 | 7.74 | 2,344 |
| E4B-it | q4 | 1.24 | 1,949 | 4.11 | 3,370 |
| E4B-it | q4 | 1.28 | 1,983 | 4.03 | 3,369 |
| E4B-it | q4f16 | 1.24 | 2,001 | 4.00 | 3,996 |
| E4B-it | q4f16 | 1.28 | 1,996 | 4.01 | 3,987 |

## Delta (1.28 vs 1.24)

| Variant | tok/s change | ms/prompt change | RSS change |
|---------|--------------|------------------|------------|
| E2B q4 | **+11%** | −100 ms | ~0 |
| E2B q4f16 | −9% | +96 ms | ~0 |
| E4B q4 | ~0% | +34 ms | ~0 |
| E4B q4f16 | ~0% | −5 ms | ~0 |

## Takeaways

- **E2B q4** is the only clear win: ~11% higher throughput, ~10% lower latency.
- **E4B** (both quants) and **E2B q4f16** are effectively unchanged on this VM; q4f16 E2B is slightly slower on 1.28 (within run-to-run noise).
- **Peak RSS** unchanged — ORT 1.28 does not reduce memory for these models.
- Quality scores were not re-run on 1.28; runtime swap should not change logits for identical ORT graph execution.

Raw JSON: `results/benchmark-gemma4-speed-ort128.json`, baseline: `results/benchmark-gemma4-speed-4variants.json`.

To reproduce 1.28: `npm run build:ort`, add `package.json` overrides to `file:vendor/onnxruntime/js/{node,common,web}`, `ONNXRUNTIME_NODE_INSTALL=skip npm install`, then `node scripts/benchmark-gemma4.mjs --variants E2B-it:q4,E2B-it:q4f16,E4B-it:q4,E4B-it:q4f16`.
