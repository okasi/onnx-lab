# Embedding benchmark agents

This repository benchmarks multilingual ONNX embedding models on **Swedish** and **Turkish** long-form text using **Transformers.js** with the **WASM** execution provider in **Node.js only** (no browser, no Python).

## Quick start

```bash
npm install
npm run generate:corpus    # optional — corpus is committed under data/
npm run benchmark:quick    # smoke run (2 models, 5 docs, q4+int8)
npm run benchmark:smoke    # all models, 3 docs, q4+int8+q4f16 (Jina)
npm run benchmark          # full run (all models & quants, 54 docs)
```

Full runs download ONNX weights from Hugging Face on first use and cache them under `.cache/transformers-node/`.

## Repository layout

| Path | Purpose |
|------|---------|
| `data/benchmark-corpus.json` | 54 long documents (27 SV + 27 TR): mortgage, legal, medical |
| `config/models.mjs` | Model registry with Hugging Face links |
| `lib/transformers-wasm.mjs` | Node.js WASM runtime bootstrap for Transformers.js |
| `scripts/benchmark.mjs` | Main benchmark runner |
| `scripts/generate-corpus.mjs` | Regenerates the corpus JSON |
| `results/` | Benchmark output JSON (gitignored) |

## WASM on Node.js

Transformers.js defaults to `onnxruntime-node` in Node. This project forces WASM by:

1. Injecting `onnxruntime-web` into `globalThis[Symbol.for('onnxruntime')]` **before** importing `@huggingface/transformers`.
2. Passing `device: 'auto'` and `session_options: { executionProviders: ['wasm'] }` to `pipeline('feature-extraction', ...)`.

See `lib/transformers-wasm.mjs` for the exact bootstrap.

## Benchmarked models

All models are loaded from their Hugging Face `onnx/` folders:

| Model | Hugging Face ONNX folder |
|-------|--------------------------|
| Granite Embedding 278M Multilingual | https://huggingface.co/sirasagi62/granite-embedding-278m-multilingual-ONNX/tree/main/onnx |
| GTE Multilingual Base | https://huggingface.co/onnx-community/gte-multilingual-base/tree/main/onnx |
| Jina Embeddings v5 Omni Nano (**text**) | https://huggingface.co/onnx-community/jina-embeddings-v5-omni-nano-ONNX/tree/main/onnx |
| Granite Embedding 311M Multilingual R2 | https://huggingface.co/onnx-community/granite-embedding-311m-multilingual-r2-ONNX/tree/main/onnx |
| EmbeddingGemma 300M | https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/tree/main/onnx |
| Qwen3 Embedding 0.6B | https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/tree/main/onnx |
| BGE-M3 | https://huggingface.co/onnx-community/bge-m3-ONNX/tree/main/onnx |
| Snowflake Arctic Embed L v2.0 | https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0/tree/main/onnx |
| Snowflake Arctic Embed M v2.0 | https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/tree/main/onnx |

### Quantization variants

For each model, the benchmark tests these quants by default:

`bnb4`, `int8`, `q4`, `q4f16`, `q8` (ONNX file `model_quantized.onnx`, alias **quantized**), `uint8`

| dtype | ONNX suffix |
|-------|-------------|
| fp32 | `model.onnx` |
| int8 | `model_int8.onnx` |
| uint8 | `model_uint8.onnx` |
| q8 / quantized | `model_quantized.onnx` |
| q4 | `model_q4.onnx` |
| q4f16 | `model_q4f16.onnx` |
| bnb4 | `model_bnb4.onnx` |

**Special cases**

- **Jina v5 Omni Nano**: text encoder only — `model_file_name: 'text_model'` (e.g. `text_model_q4f16.onnx`).
- **Arctic Embed L v2.0 O4**: extra variant via `model_file_name: 'model_O4'` (see `config/models.mjs`).
- **fp32**: often requires large external `.onnx_data` shards. Not included in default benchmark quants.

## Corpus

- **54 documents** (≥50 required): 9 Swedish + 9 Turkish per topic.
- Topics: `mortgage`, `legal`, `medical`.
- Each document is a multi-paragraph domain text (~150–250 words).
- **27 cross-lingual query pairs** pair Swedish/Turkish documents on the same topic for cosine-similarity sanity checks.

Regenerate:

```bash
npm run generate:corpus
```

## Benchmark CLI

```bash
node scripts/benchmark.mjs [options]
```

| Flag | Description |
|------|-------------|
| `--quick` | 2 models, 5 documents, skip fp32 |
| `--include-fp32` | Include fp32 (large downloads) |
| `--max-texts N` | Embed only the first N documents |
| `--model id[,id]` | Restrict to specific Hugging Face model ids |
| `--dtype d[,d]` | Restrict to specific quants |
| `--output path` | Custom results JSON path |

### Metrics recorded

Per model × quant variant:

- **Timing**: `load_time_ms`, `total_time_ms`, per-document latency (mean / p50 / p95 / total)
- **Memory**: `memory.peak_rss_mb`, `peak_heap_used_mb`, `peak_external_mb`
- **Quality**:
  - `topic_cohesion_mean` / `topic_separation_mean` / `topic_discrimination`
  - `cross_lingual_pairs.mean_cosine` (SV↔TR same-topic pairs)
  - `retrieval.recall_at_1`, `recall_at_5`, `recall_at_10`
  - `composite_score` (weighted blend of cross-lingual, discrimination, recall@5)

Run-level summary includes wall time, peak RSS, leaderboard, and best variant per model.

## Runtime & crash resilience

Each variant runs in an **isolated subprocess** (`scripts/benchmark-variant.mjs`) so OOM kills do not crash the full benchmark.

**Backend strategy (`auto`):**
1. Try **WASM** first (onnxruntime-web)
2. On failure (external data, GatherBlockQuantized, OOM), automatically retry **CPU** (onnxruntime-node)

Models with known WASM issues use `backend: 'cpu'` directly (Jina text, EmbeddingGemma).

### EmbeddingGemma 300M

WASM cannot load `.onnx_data` shards. On CPU all quants work:

| Quant | Backend | Notes |
|-------|---------|-------|
| q4 | cpu (auto fallback) | Standard `model_q4.onnx` + data |
| no_gather_q4 | cpu | `model_no_gather_q4.onnx` — avoids GatherBlockQuantized |
| quantized (q8) | cpu | `model_quantized.onnx` + data |

Example full-corpus result (54 docs): quality **~0.63**, cross-lingual cosine **~0.81**, XL-R@5 **0.72**.

## Retrieval metrics (robust)

| Metric | Meaning |
|--------|---------|
| **R@5** (`topic_any`) | Same topic in top 5, any language |
| **XL-R@5** (`topic_cross_lang`) | Same topic in top 5, **different** language (SV↔TR) — harder, more meaningful for this benchmark |
| **MRR@10** | Mean reciprocal rank of first relevant hit |
| **XLing** | Mean cosine of paired SV/TR documents on identical topics |

Composite score weights: cross-lingual cosine (35%), topic discrimination (25%), XL-R@5 (25%), R@5 (15%).

Record failures instead of hiding them:

| Issue | Typical symptom |
|-------|-----------------|
| External data (`*.onnx_data`) missing | fp32 init error for split ONNX graphs |
| `q4f16` dtype mismatch | ORT WASM float16 vs float32 tensor error |
| Jina architecture | Warning: `JinaEmbeddingsV5OmniModel` not in MODEL_TYPE_MAPPING; text encoder uses `text_model_*.onnx` only |
| EmbeddingGemma q4 | `GatherBlockQuantized` not implemented in ORT WASM |
| O4 variant | Non-standard filename; requires `model_file_name` override |

## Agent workflow

When extending this repo:

1. **Add a model** — edit `config/models.mjs` with `id`, `name`, `url`, and any `model_file_name` / `extra_variants`.
2. **Regenerate corpus** only if benchmark text requirements change.
3. **Run quick benchmark** after code changes: `npm run benchmark:quick`.
4. **Run full benchmark** before publishing results: `npm run benchmark` (long-running; downloads multi-GB weights).
5. **Commit** `data/benchmark-corpus.json` and code; keep `results/` and `.cache/` out of git.
6. **Document** new quirks in this file under "Known WASM limitations".

## References

- Transformers.js: https://huggingface.co/docs/transformers.js
- Transformers.js feature extraction: https://huggingface.co/docs/transformers.js/api/pipelines#module_pipelines.FeatureExtractionPipeline
- ONNX embedding models on Hub: https://huggingface.co/models?pipeline_tag=feature-extraction&library=transformers.js
- `ModelRegistry` API: https://huggingface.co/docs/transformers.js/api/utils/model_registry
- ONNX Runtime Web (WASM): https://onnxruntime.ai/docs/execution-providers/Web-Execution-Provider.html
- Swedish mortgage supervision (FI): https://www.fi.se/
- Turkish banking regulation (BDDK): https://www.bddk.org.tr/

## Environment

- **Node.js** 18+ (tested on 22.x)
- **@huggingface/transformers** 4.x
- Network access for first-time model download
- Disk space: several GB for all quants across all models
