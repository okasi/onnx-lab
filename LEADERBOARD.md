# ONNX Lab Leaderboard

Swedish/Turkish corpus (54 documents) · Transformers.js · Node.js WASM with CPU fallback · [onnx-lab](https://github.com/okasi/onnx-lab)

## Run summary

| Field | Value |
|-------|-------|
| Generated | 2026-06-17 |
| Source | `benchmark-1781714103604.json` |
| Wall time | 39m 48s |
| Peak RSS | 6611.5 MB |
| Documents | 54 |
| Variants tested | 29 |
| Succeeded | 25 |
| Failed | 4 |

### Metric definitions

| Metric | Description |
|--------|-------------|
| **Quality** | Composite: XLing (35%) + topic discrimination (25%) + XL-R@5 (25%) + R@5 (15%) |
| **XLing** | Mean cosine similarity of paired SV↔TR documents (same topic) |
| **XL-R@5** | Cross-lingual recall@5: query in one language, relevant doc in the *other* language in top 5 |
| **R@1 / R@3 / R@5 / R@10** | Same-topic retrieval recall (any language) |
| **MRR@10** | Mean reciprocal rank of first cross-lingual same-topic hit |
| **Cohesion / Separation** | Mean cosine within-topic vs between-topic |
| **RSS** | Peak resident set size during variant run |

---

## Ranked leaderboard (successful variants)

| Rank | Model | Quant | Backend | Quality | XLing | XL-R@5 | R@5 | R@3 | R@1 | MRR@10 | Cohesion | Sep. | ms/doc | Total | RSS MB | Dim |
|------|-------|-------|---------|---------|-------|--------|-----|-----|-----|--------|----------|------|--------|-------|--------|-----|
| 1 | GTE Multilingual Base | bnb4 | wasm | 0.7298 | 0.9527 | 0.9630 | 0.9815 | 0.9630 | 0.8889 | 0.7906 | 0.9205 | 0.8869 | 927.7 | 55s | 4909.4 | 768 |
| 2 | BGE-M3 | bnb4 | wasm | 0.7263 | 0.8829 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 0.8340 | 0.8012 | 0.7320 | 2944.9 | 2m 47s | 6611.5 | 1024 |
| 3 | BGE-M3 | q4 | wasm | 0.7263 | 0.8829 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 0.8340 | 0.8012 | 0.7320 | 2904.8 | 2m 44s | 5717.8 | 1024 |
| 4 | GTE Multilingual Base | q4 | wasm | 0.7252 | 0.9473 | 0.9444 | 0.9815 | 0.9630 | 0.9074 | 0.7190 | 0.9150 | 0.8738 | 1078.4 | 1m 3s | 5035.8 | 768 |
| 5 | BGE-M3 | int8 | wasm | 0.7250 | 0.8839 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 0.8056 | 0.7999 | 0.7374 | 1069.5 | 1m 2s | 2690.8 | 1024 |
| 6 | BGE-M3 | quantized (q8) | wasm | 0.7250 | 0.8839 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 0.8056 | 0.7999 | 0.7374 | 1062.3 | 1m 0s | 3151.3 | 1024 |
| 7 | BGE-M3 | uint8 | wasm | 0.7242 | 0.8825 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 0.8488 | 0.8005 | 0.7393 | 1063.2 | 1m 1s | 2125.5 | 1024 |
| 8 | GTE Multilingual Base | uint8 | wasm | 0.6959 | 0.9489 | 0.8333 | 0.9815 | 0.9815 | 0.8704 | 0.6297 | 0.9235 | 0.8906 | 415.7 | 25s | 2108.7 | 768 |
| 9 | GTE Multilingual Base | int8 | wasm | 0.6936 | 0.9487 | 0.8333 | 0.9630 | 0.9444 | 0.8148 | 0.6138 | 0.9218 | 0.8868 | 430.1 | 25s | 1938.6 | 768 |
| 10 | GTE Multilingual Base | quantized (q8) | wasm | 0.6936 | 0.9487 | 0.8333 | 0.9630 | 0.9444 | 0.8148 | 0.6138 | 0.9218 | 0.8868 | 415.3 | 25s | 2191.7 | 768 |
| 11 | EmbeddingGemma 300M | no_gather_q4 | cpu | 0.6363 | 0.8165 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 0.5818 | 0.7277 | 0.6477 | 290.8 | 17s | 1510.3 | 768 |
| 12 | EmbeddingGemma 300M | q4f16 | cpu | 0.6362 | 0.8161 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 0.5698 | 0.7277 | 0.6477 | 91.7 | 6s | 585.2 | 768 |
| 13 | EmbeddingGemma 300M | q4 | cpu | 0.6361 | 0.8159 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 0.5605 | 0.7275 | 0.6475 | 80.3 | 5s | 574.8 | 768 |
| 14 | EmbeddingGemma 300M | quantized (q8) | cpu | 0.6326 | 0.8045 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 0.5771 | 0.7085 | 0.6266 | 206.8 | 12s | 1603.0 | 768 |
| 15 | Granite Embedding 278M Multilingual | uint8 | wasm | 0.6258 | 0.7948 | 0.7037 | 1.0000 | 0.9444 | 0.8333 | 0.5859 | 0.7198 | 0.6329 | 316.2 | 19s | 1869.7 | 768 |
| 16 | Granite Embedding 278M Multilingual | bnb4 | wasm | 0.6178 | 0.7776 | 0.7037 | 0.9815 | 0.9444 | 0.8148 | 0.5953 | 0.6983 | 0.6084 | 695.6 | 44s | 3174.6 | 768 |
| 17 | Granite Embedding 278M Multilingual | int8 | wasm | 0.6174 | 0.7927 | 0.6852 | 0.9815 | 0.9444 | 0.7593 | 0.5850 | 0.7180 | 0.6323 | 310.9 | 19s | 1687.2 | 768 |
| 18 | Granite Embedding 278M Multilingual | quantized (q8) | wasm | 0.6174 | 0.7927 | 0.6852 | 0.9815 | 0.9444 | 0.7593 | 0.5850 | 0.7180 | 0.6323 | 327.8 | 20s | 1581.9 | 768 |
| 19 | Granite Embedding 278M Multilingual | q4 | wasm | 0.6135 | 0.7819 | 0.6667 | 1.0000 | 0.9630 | 0.8148 | 0.5844 | 0.7054 | 0.6129 | 805.3 | 47s | 4680.8 | 768 |
| 20 | Qwen3 Embedding 0.6B | q4 | wasm | 0.5034 | 0.7679 | 0.2778 | 0.9815 | 0.9444 | 0.8889 | 0.2137 | 0.7594 | 0.6873 | 5590.9 | 5m 7s | 3584.6 | 1024 |
| 21 | Qwen3 Embedding 0.6B | q4f16 | wasm | 0.5034 | 0.7678 | 0.2778 | 0.9815 | 0.9444 | 0.8889 | 0.2137 | 0.7594 | 0.6874 | 5816.4 | 5m 18s | 2188.7 | 1024 |
| 22 | Qwen3 Embedding 0.6B | bnb4 | wasm | 0.4782 | 0.7336 | 0.2222 | 0.9815 | 0.9630 | 0.8704 | 0.1331 | 0.7370 | 0.6625 | 7705.6 | 7m 1s | 3127.1 | 1024 |
| 23 | Qwen3 Embedding 0.6B | int8 | wasm | 0.4199 | 0.7195 | 0.0370 | 0.9815 | 0.9630 | 0.8519 | 0.0235 | 0.7690 | 0.7226 | 2262.5 | 2m 6s | 2198.3 | 1024 |
| 24 | Qwen3 Embedding 0.6B | quantized (q8) | wasm | 0.4082 | 0.7138 | 0.0000 | 0.9815 | 0.9630 | 0.8148 | 0.0161 | 0.7700 | 0.7255 | 2266.2 | 2m 6s | 2328.2 | 1024 |
| 25 | Qwen3 Embedding 0.6B | uint8 | wasm | 0.4082 | 0.7138 | 0.0000 | 0.9815 | 0.9630 | 0.8148 | 0.0161 | 0.7700 | 0.7255 | 2263.1 | 2m 6s | 2227.3 | 1024 |

---

## Full results (all variants)

| Model | Quant | Status | Backend | Dim | Load | Total | ms/doc | p95 | RSS MB | Quality | XLing | XL-R@5 | R@5 | R@3 | R@1 | R@10 | Cohesion | Separation | Error |
|-------|-------|--------|---------|-----|------|-------|--------|-----|--------|---------|-------|--------|-----|-----|-----|------|----------|------------|-------|
| BGE-M3 | bnb4 | ok | wasm | 1024 | 8.0s | 2m 47s | 2944.9 | — | 6611.5 | 0.7263 | 0.8829 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.8012 | 0.7320 | — |
| BGE-M3 | int8 | ok | wasm | 1024 | 4.5s | 1m 2s | 1069.5 | — | 2690.8 | 0.7250 | 0.8839 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.7999 | 0.7374 | — |
| BGE-M3 | q4 | ok | wasm | 1024 | 7.5s | 2m 44s | 2904.8 | — | 5717.8 | 0.7263 | 0.8829 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.8012 | 0.7320 | — |
| BGE-M3 | q4f16 | error | cpu | — | 1.0s | 1s | — | — | 173.4 | — | — | — | — | — | — | — | — | — | FP16 tensor type mismatch |
| BGE-M3 | quantized (q8) | ok | wasm | 1024 | 3.3s | 1m 0s | 1062.3 | — | 3151.3 | 0.7250 | 0.8839 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.7999 | 0.7374 | — |
| BGE-M3 | uint8 | ok | wasm | 1024 | 3.4s | 1m 1s | 1063.2 | — | 2125.5 | 0.7242 | 0.8825 | 1.0000 | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.8005 | 0.7393 | — |
| EmbeddingGemma 300M | q4 | ok | cpu | 768 | 1.3s | 5s | 80.3 | — | 574.8 | 0.6361 | 0.8159 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 1.0000 | 0.7275 | 0.6475 | — |
| EmbeddingGemma 300M | q4f16 | ok | cpu | 768 | 1.3s | 6s | 91.7 | — | 585.2 | 0.6362 | 0.8161 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 1.0000 | 0.7277 | 0.6477 | — |
| EmbeddingGemma 300M | quantized (q8) | ok | cpu | 768 | 1.1s | 12s | 206.8 | — | 1603.0 | 0.6326 | 0.8045 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 1.0000 | 0.7085 | 0.6266 | — |
| EmbeddingGemma 300M | no_gather_q4 | ok | cpu | 768 | 1.6s | 17s | 290.8 | — | 1510.3 | 0.6363 | 0.8165 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 1.0000 | 0.7277 | 0.6477 | — |
| GTE Multilingual Base | bnb4 | ok | wasm | 768 | 5.6s | 55s | 927.7 | — | 4909.4 | 0.7298 | 0.9527 | 0.9630 | 0.9815 | 0.9630 | 0.8889 | 1.0000 | 0.9205 | 0.8869 | — |
| GTE Multilingual Base | int8 | ok | wasm | 768 | 2.5s | 25s | 430.1 | — | 1938.6 | 0.6936 | 0.9487 | 0.8333 | 0.9630 | 0.9444 | 0.8148 | 1.0000 | 0.9218 | 0.8868 | — |
| GTE Multilingual Base | q4 | ok | wasm | 768 | 5.4s | 1m 3s | 1078.4 | — | 5035.8 | 0.7252 | 0.9473 | 0.9444 | 0.9815 | 0.9630 | 0.9074 | 1.0000 | 0.9150 | 0.8738 | — |
| GTE Multilingual Base | q4f16 | error | cpu | — | 1.0s | 0s | — | — | 149.1 | — | — | — | — | — | — | — | — | — | FP16 graph fusion error |
| GTE Multilingual Base | quantized (q8) | ok | wasm | 768 | 2.7s | 25s | 415.3 | — | 2191.7 | 0.6936 | 0.9487 | 0.8333 | 0.9630 | 0.9444 | 0.8148 | 1.0000 | 0.9218 | 0.8868 | — |
| GTE Multilingual Base | uint8 | ok | wasm | 768 | 2.6s | 25s | 415.7 | — | 2108.7 | 0.6959 | 0.9489 | 0.8333 | 0.9815 | 0.9815 | 0.8704 | 1.0000 | 0.9235 | 0.8906 | — |
| Granite Embedding 278M Multilingual | bnb4 | ok | wasm | 768 | 6.4s | 44s | 695.6 | — | 3174.6 | 0.6178 | 0.7776 | 0.7037 | 0.9815 | 0.9444 | 0.8148 | 1.0000 | 0.6983 | 0.6084 | — |
| Granite Embedding 278M Multilingual | int8 | ok | wasm | 768 | 2.2s | 19s | 310.9 | — | 1687.2 | 0.6174 | 0.7927 | 0.6852 | 0.9815 | 0.9444 | 0.7593 | 1.0000 | 0.7180 | 0.6323 | — |
| Granite Embedding 278M Multilingual | q4 | ok | wasm | 768 | 4.2s | 47s | 805.3 | — | 4680.8 | 0.6135 | 0.7819 | 0.6667 | 1.0000 | 0.9630 | 0.8148 | 1.0000 | 0.7054 | 0.6129 | — |
| Granite Embedding 278M Multilingual | q4f16 | error | cpu | — | 0.9s | 0s | — | — | 186.1 | — | — | — | — | — | — | — | — | — | FP16 graph fusion error |
| Granite Embedding 278M Multilingual | quantized (q8) | ok | wasm | 768 | 2.2s | 20s | 327.8 | — | 1581.9 | 0.6174 | 0.7927 | 0.6852 | 0.9815 | 0.9444 | 0.7593 | 1.0000 | 0.7180 | 0.6323 | — |
| Granite Embedding 278M Multilingual | uint8 | ok | wasm | 768 | 2.8s | 19s | 316.2 | — | 1869.7 | 0.6258 | 0.7948 | 0.7037 | 1.0000 | 0.9444 | 0.8333 | 1.0000 | 0.7198 | 0.6329 | — |
| Jina Embeddings v5 Omni Nano (text) | q4f16 | error | cpu | — | 0.2s | 0s | — | — | 112.6 | — | — | — | — | — | — | — | — | — | Missing .onnx_data shard |
| Qwen3 Embedding 0.6B | bnb4 | ok | wasm | 1024 | 5.7s | 7m 1s | 7705.6 | — | 3127.1 | 0.4782 | 0.7336 | 0.2222 | 0.9815 | 0.9630 | 0.8704 | 1.0000 | 0.7370 | 0.6625 | — |
| Qwen3 Embedding 0.6B | int8 | ok | wasm | 1024 | 4.0s | 2m 6s | 2262.5 | — | 2198.3 | 0.4199 | 0.7195 | 0.0370 | 0.9815 | 0.9630 | 0.8519 | 1.0000 | 0.7690 | 0.7226 | — |
| Qwen3 Embedding 0.6B | q4 | ok | wasm | 1024 | 5.4s | 5m 7s | 5590.9 | — | 3584.6 | 0.5034 | 0.7679 | 0.2778 | 0.9815 | 0.9444 | 0.8889 | 1.0000 | 0.7594 | 0.6873 | — |
| Qwen3 Embedding 0.6B | q4f16 | ok | wasm | 1024 | 4.1s | 5m 18s | 5816.4 | — | 2188.7 | 0.5034 | 0.7678 | 0.2778 | 0.9815 | 0.9444 | 0.8889 | 1.0000 | 0.7594 | 0.6874 | — |
| Qwen3 Embedding 0.6B | quantized (q8) | ok | wasm | 1024 | 4.1s | 2m 6s | 2266.2 | — | 2328.2 | 0.4082 | 0.7138 | 0.0000 | 0.9815 | 0.9630 | 0.8148 | 1.0000 | 0.7700 | 0.7255 | — |
| Qwen3 Embedding 0.6B | uint8 | ok | wasm | 1024 | 4.0s | 2m 6s | 2263.1 | — | 2227.3 | 0.4082 | 0.7138 | 0.0000 | 0.9815 | 0.9630 | 0.8148 | 1.0000 | 0.7700 | 0.7255 | — |

---

## Best variant per model

| Model | Best quant | Quality | XLing | XL-R@5 | R@5 | R@3 | ms/doc | RSS MB | Backend |
|-------|------------|---------|-------|--------|-----|-----|--------|--------|---------|
| BGE-M3 | bnb4 | 0.7263 | 0.8829 | 1.0000 | 1.0000 | 1.0000 | 2944.9 | 6611.5 | wasm |
| EmbeddingGemma 300M | no_gather_q4 | 0.6363 | 0.8165 | 0.7222 | 1.0000 | 0.9630 | 290.8 | 1510.3 | cpu |
| GTE Multilingual Base | bnb4 | 0.7298 | 0.9527 | 0.9630 | 0.9815 | 0.9630 | 927.7 | 4909.4 | wasm |
| Granite Embedding 278M Multilingual | uint8 | 0.6258 | 0.7948 | 0.7037 | 1.0000 | 0.9444 | 316.2 | 1869.7 | wasm |
| Jina Embeddings v5 Omni Nano (text) | — | — | — | — | — | — | — | — | all failed |
| Qwen3 Embedding 0.6B | q4 | 0.5034 | 0.7679 | 0.2778 | 0.9815 | 0.9444 | 5590.9 | 3584.6 | wasm |

---

## Failure summary

| Error | Count | Variants |
|-------|-------|----------|
| FP16 graph fusion error | 2 | Granite Embedding 278M Multilingual / q4f16; GTE Multilingual Base / q4f16 |
| Missing .onnx_data shard | 1 | Jina Embeddings v5 Omni Nano (text) / q4f16 |
| FP16 tensor type mismatch | 1 | BGE-M3 / q4f16 |

---

*Regenerate: `npm run leaderboard`*
