# Embedding Benchmark Leaderboard

Swedish/Turkish corpus (54 documents) · Transformers.js · Node.js WASM with CPU fallback

## Run summary

| Field | Value |
|-------|-------|
| Generated | 2026-06-17 |
| Source | `benchmark-full-1781706134847.json` + `benchmark-1781707883726.json` |
| Wall time | 48m 48s |
| Peak RSS | 12023 MB |
| Documents | 54 |
| Variants tested | 28 |
| Succeeded | 22 |
| Failed | 6 |

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
| 1 | GTE Multilingual Base | q4 | wasm | 0.6857 | 0.9473 | — | 0.9815 | 0.9630 | 0.9074 | — | 0.9150 | 0.8738 | 1099.2 | 1m 4s | 6396.9 | 768 |
| 2 | GTE Multilingual Base | bnb4 | wasm | 0.6856 | 0.9527 | — | 0.9815 | 0.9630 | 0.8889 | — | 0.9205 | 0.8869 | 957.3 | 2m 44s | 6126.9 | 768 |
| 3 | GTE Multilingual Base | uint8 | wasm | 0.6839 | 0.9489 | — | 0.9815 | 0.9815 | 0.8704 | — | 0.9235 | 0.8906 | 409.6 | 32s | 5132.4 | 768 |
| 4 | GTE Multilingual Base | int8 | wasm | 0.6789 | 0.9487 | — | 0.9630 | 0.9444 | 0.8148 | — | 0.9218 | 0.8868 | 410.1 | 23s | 5661.2 | 768 |
| 5 | GTE Multilingual Base | quantized (q8) | wasm | 0.6789 | 0.9487 | — | 0.9630 | 0.9444 | 0.8148 | — | 0.9218 | 0.8868 | 406.1 | 25s | 5324.4 | 768 |
| 6 | BGE-M3 | int8 | wasm | 0.6723 | 0.8839 | — | 1.0000 | 1.0000 | 0.9259 | — | 0.7999 | 0.7374 | 1064.0 | 1m 1s | 8369.7 | 1024 |
| 7 | BGE-M3 | quantized (q8) | wasm | 0.6723 | 0.8839 | — | 1.0000 | 1.0000 | 0.9259 | — | 0.7999 | 0.7374 | 1078.3 | 1m 0s | 9930.6 | 1024 |
| 8 | BGE-M3 | uint8 | wasm | 0.6714 | 0.8825 | — | 1.0000 | 1.0000 | 0.9259 | — | 0.8005 | 0.7393 | 1061.3 | 1m 1s | 10532.5 | 1024 |
| 9 | Granite Embedding 278M Multilingual | uint8 | wasm | 0.6440 | 0.7948 | — | 1.0000 | 0.9444 | 0.8333 | — | 0.7198 | 0.6329 | 309.7 | 24s | 4924.4 | 768 |
| 10 | Granite Embedding 278M Multilingual | q4 | wasm | 0.6405 | 0.7819 | — | 1.0000 | 0.9630 | 0.8148 | — | 0.7054 | 0.6129 | 836.1 | 50s | 6073.5 | 768 |
| 11 | Granite Embedding 278M Multilingual | int8 | wasm | 0.6373 | 0.7927 | — | 0.9815 | 0.9444 | 0.7593 | — | 0.7180 | 0.6323 | 309.6 | 18s | 4653.8 | 768 |
| 12 | Granite Embedding 278M Multilingual | quantized (q8) | wasm | 0.6373 | 0.7927 | — | 0.9815 | 0.9444 | 0.7593 | — | 0.7180 | 0.6323 | 308.6 | 24s | 5173.2 | 768 |
| 13 | EmbeddingGemma 300M | no_gather_q4 | cpu | 0.6363 | 0.8165 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 0.5818 | 0.7277 | 0.6477 | 316.4 | 18s | 1510.4 | 768 |
| 14 | EmbeddingGemma 300M | q4 | cpu | 0.6361 | 0.8159 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 0.5605 | 0.7275 | 0.6475 | 83.7 | 5s | 579.8 | 768 |
| 15 | EmbeddingGemma 300M | quantized (q8) | cpu | 0.6326 | 0.8045 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 0.5771 | 0.7085 | 0.6266 | 254.5 | 14s | 1600.7 | 768 |
| 16 | Granite Embedding 278M Multilingual | bnb4 | wasm | 0.6324 | 0.7776 | — | 0.9815 | 0.9444 | 0.8148 | — | 0.6983 | 0.6084 | 746.8 | 1m 9s | 3634.8 | 768 |
| 17 | Qwen3 Embedding 0.6B | q4 | wasm | 0.6232 | 0.7679 | — | 0.9815 | 0.9444 | 0.8889 | — | 0.7594 | 0.6873 | 5518.5 | 5m 2s | 6436.6 | 1024 |
| 18 | Qwen3 Embedding 0.6B | q4f16 | wasm | 0.6232 | 0.7678 | — | 0.9815 | 0.9444 | 0.8889 | — | 0.7594 | 0.6874 | 5818.5 | 5m 47s | 6313.5 | 1024 |
| 19 | Qwen3 Embedding 0.6B | bnb4 | wasm | 0.6103 | 0.7336 | — | 0.9815 | 0.9630 | 0.8704 | — | 0.7370 | 0.6625 | 5007.7 | 4m 44s | 6382.2 | 1024 |
| 20 | Qwen3 Embedding 0.6B | int8 | wasm | 0.5962 | 0.7195 | — | 0.9815 | 0.9630 | 0.8519 | — | 0.7690 | 0.7226 | 2261.4 | 2m 5s | 5750.8 | 1024 |
| 21 | Qwen3 Embedding 0.6B | quantized (q8) | wasm | 0.5933 | 0.7138 | — | 0.9815 | 0.9630 | 0.8148 | — | 0.7700 | 0.7255 | 2275.3 | 2m 21s | 6206.5 | 1024 |
| 22 | Qwen3 Embedding 0.6B | uint8 | wasm | 0.5933 | 0.7138 | — | 0.9815 | 0.9630 | 0.8148 | — | 0.7700 | 0.7255 | 2271.1 | 2m 24s | 6285.3 | 1024 |

---

## Full results (all variants)

| Model | Quant | Status | Backend | Dim | Load | Total | ms/doc | p95 | RSS MB | Quality | XLing | XL-R@5 | R@5 | R@3 | R@1 | R@10 | Cohesion | Separation | Error |
|-------|-------|--------|---------|-----|------|-------|--------|-----|--------|---------|-------|--------|-----|-----|-----|------|----------|------------|-------|
| BGE-M3 | bnb4 | error | — | — | 7.4s | 7s | — | — | 9266.6 | — | — | — | — | — | — | — | — | — | OOM (std::bad_alloc) |
| BGE-M3 | int8 | ok | wasm | 1024 | 3.6s | 1m 1s | 1064.0 | 1906.0 | 8369.7 | 0.6723 | 0.8839 | — | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.7999 | 0.7374 | — |
| BGE-M3 | q4 | error | — | — | 6.4s | 6s | — | — | 10967.8 | — | — | — | — | — | — | — | — | — | OOM (std::bad_alloc) |
| BGE-M3 | q4f16 | error | — | — | 2.5s | 2s | — | — | 10949.5 | — | — | — | — | — | — | — | — | — | FP16 tensor type mismatch |
| BGE-M3 | quantized (q8) | ok | wasm | 1024 | 2.5s | 1m 0s | 1078.3 | 1892.8 | 9930.6 | 0.6723 | 0.8839 | — | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.7999 | 0.7374 | — |
| BGE-M3 | uint8 | ok | wasm | 1024 | 3.9s | 1m 1s | 1061.3 | 1908.3 | 10532.5 | 0.6714 | 0.8825 | — | 1.0000 | 1.0000 | 0.9259 | 1.0000 | 0.8005 | 0.7393 | — |
| EmbeddingGemma 300M | q4 | ok | cpu | 768 | 1.3s | 5s | 83.7 | — | 579.8 | 0.6361 | 0.8159 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 1.0000 | 0.7275 | 0.6475 | — |
| EmbeddingGemma 300M | quantized (q8) | ok | cpu | 768 | 1.1s | 14s | 254.5 | — | 1600.7 | 0.6326 | 0.8045 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 1.0000 | 0.7085 | 0.6266 | — |
| EmbeddingGemma 300M | no_gather_q4 | ok | cpu | 768 | 1.5s | 18s | 316.4 | — | 1510.4 | 0.6363 | 0.8165 | 0.7222 | 1.0000 | 0.9630 | 0.9074 | 1.0000 | 0.7277 | 0.6477 | — |
| GTE Multilingual Base | bnb4 | ok | wasm | 768 | 112.9s | 2m 44s | 957.3 | 1636.0 | 6126.9 | 0.6856 | 0.9527 | — | 0.9815 | 0.9630 | 0.8889 | 1.0000 | 0.9205 | 0.8869 | — |
| GTE Multilingual Base | int8 | ok | wasm | 768 | 1.6s | 23s | 410.1 | 737.6 | 5661.2 | 0.6789 | 0.9487 | — | 0.9630 | 0.9444 | 0.8148 | 1.0000 | 0.9218 | 0.8868 | — |
| GTE Multilingual Base | q4 | ok | wasm | 768 | 5.2s | 1m 4s | 1099.2 | 1830.8 | 6396.9 | 0.6857 | 0.9473 | — | 0.9815 | 0.9630 | 0.9074 | 1.0000 | 0.9150 | 0.8738 | — |
| GTE Multilingual Base | q4f16 | error | — | — | 15.3s | 15s | — | — | 6368.7 | — | — | — | — | — | — | — | — | — | FP16 graph fusion error |
| GTE Multilingual Base | quantized (q8) | ok | wasm | 768 | 3.0s | 25s | 406.1 | 732.1 | 5324.4 | 0.6789 | 0.9487 | — | 0.9630 | 0.9444 | 0.8148 | 1.0000 | 0.9218 | 0.8868 | — |
| GTE Multilingual Base | uint8 | ok | wasm | 768 | 10.0s | 32s | 409.6 | 761.8 | 5132.4 | 0.6839 | 0.9489 | — | 0.9815 | 0.9815 | 0.8704 | 1.0000 | 0.9235 | 0.8906 | — |
| Granite Embedding 278M Multilingual | bnb4 | ok | wasm | 768 | 29.1s | 1m 9s | 746.8 | 1204.4 | 3634.8 | 0.6324 | 0.7776 | — | 0.9815 | 0.9444 | 0.8148 | 1.0000 | 0.6983 | 0.6084 | — |
| Granite Embedding 278M Multilingual | int8 | ok | wasm | 768 | 1.6s | 18s | 309.6 | 566.1 | 4653.8 | 0.6373 | 0.7927 | — | 0.9815 | 0.9444 | 0.7593 | 1.0000 | 0.7180 | 0.6323 | — |
| Granite Embedding 278M Multilingual | q4 | ok | wasm | 768 | 4.7s | 50s | 836.1 | 1385.8 | 6073.5 | 0.6405 | 0.7819 | — | 1.0000 | 0.9630 | 0.8148 | 1.0000 | 0.7054 | 0.6129 | — |
| Granite Embedding 278M Multilingual | q4f16 | error | — | — | 47.9s | 47s | — | — | 6053.0 | — | — | — | — | — | — | — | — | — | FP16 graph fusion error |
| Granite Embedding 278M Multilingual | quantized (q8) | ok | wasm | 768 | 8.1s | 24s | 308.6 | 564.3 | 5173.2 | 0.6373 | 0.7927 | — | 0.9815 | 0.9444 | 0.7593 | 1.0000 | 0.7180 | 0.6323 | — |
| Granite Embedding 278M Multilingual | uint8 | ok | wasm | 768 | 7.4s | 24s | 309.7 | 566.2 | 4924.4 | 0.6440 | 0.7948 | — | 1.0000 | 0.9444 | 0.8333 | 1.0000 | 0.7198 | 0.6329 | — |
| Jina Embeddings v5 Omni Nano (text) | q4f16 | error | — | — | 0.3s | 0s | — | — | 4475.5 | — | — | — | — | — | — | — | — | — | Missing .onnx_data shard |
| Qwen3 Embedding 0.6B | bnb4 | ok | wasm | 1024 | 14.2s | 4m 44s | 5007.7 | 9089.4 | 6382.2 | 0.6103 | 0.7336 | — | 0.9815 | 0.9630 | 0.8704 | 1.0000 | 0.7370 | 0.6625 | — |
| Qwen3 Embedding 0.6B | int8 | ok | wasm | 1024 | 3.2s | 2m 5s | 2261.4 | 4336.4 | 5750.8 | 0.5962 | 0.7195 | — | 0.9815 | 0.9630 | 0.8519 | 1.0000 | 0.7690 | 0.7226 | — |
| Qwen3 Embedding 0.6B | q4 | ok | wasm | 1024 | 4.6s | 5m 2s | 5518.5 | 9512.8 | 6436.6 | 0.6232 | 0.7679 | — | 0.9815 | 0.9444 | 0.8889 | 1.0000 | 0.7594 | 0.6873 | — |
| Qwen3 Embedding 0.6B | q4f16 | ok | wasm | 1024 | 32.8s | 5m 47s | 5818.5 | 9960.1 | 6313.5 | 0.6232 | 0.7678 | — | 0.9815 | 0.9444 | 0.8889 | 1.0000 | 0.7594 | 0.6874 | — |
| Qwen3 Embedding 0.6B | quantized (q8) | ok | wasm | 1024 | 18.3s | 2m 21s | 2275.3 | 4318.4 | 6206.5 | 0.5933 | 0.7138 | — | 0.9815 | 0.9630 | 0.8148 | 1.0000 | 0.7700 | 0.7255 | — |
| Qwen3 Embedding 0.6B | uint8 | ok | wasm | 1024 | 22.3s | 2m 24s | 2271.1 | 4275.2 | 6285.3 | 0.5933 | 0.7138 | — | 0.9815 | 0.9630 | 0.8148 | 1.0000 | 0.7700 | 0.7255 | — |

---

## Best variant per model

| Model | Best quant | Quality | XLing | XL-R@5 | R@5 | R@3 | ms/doc | RSS MB | Backend |
|-------|------------|---------|-------|--------|-----|-----|--------|--------|---------|
| BGE-M3 | int8 | 0.6723 | 0.8839 | — | 1.0000 | 1.0000 | 1064.0 | 8369.7 | wasm |
| EmbeddingGemma 300M | no_gather_q4 | 0.6363 | 0.8165 | 0.7222 | 1.0000 | 0.9630 | 316.4 | 1510.4 | cpu |
| GTE Multilingual Base | q4 | 0.6857 | 0.9473 | — | 0.9815 | 0.9630 | 1099.2 | 6396.9 | wasm |
| Granite Embedding 278M Multilingual | uint8 | 0.6440 | 0.7948 | — | 1.0000 | 0.9444 | 309.7 | 4924.4 | wasm |
| Jina Embeddings v5 Omni Nano (text) | — | — | — | — | — | — | — | — | all failed |
| Qwen3 Embedding 0.6B | q4 | 0.6232 | 0.7679 | — | 0.9815 | 0.9444 | 5518.5 | 6436.6 | wasm |

---

## Failure summary

| Error | Count | Variants |
|-------|-------|----------|
| FP16 graph fusion error | 2 | Granite Embedding 278M Multilingual / q4f16; GTE Multilingual Base / q4f16 |
| OOM (std::bad_alloc) | 2 | BGE-M3 / bnb4; BGE-M3 / q4 |
| Missing .onnx_data shard | 1 | Jina Embeddings v5 Omni Nano (text) / q4f16 |
| FP16 tensor type mismatch | 1 | BGE-M3 / q4f16 |

---

*Regenerate: `npm run leaderboard`*
