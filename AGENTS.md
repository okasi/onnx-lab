# onnx-lab

ONNX model lab for **embeddings and LLMs** with **Transformers.js** across **WASM**, **CPU**, and **WebGPU**. Includes Swedish/Turkish embedding benchmarks, quant compatibility probes, and runtime bootstrap notes.

## Quick start

```bash
npm install
npm run generate:corpus    # optional — corpus is committed under data/
npm run benchmark:quick    # smoke run (2 models, 5 docs, q4+int8)
npm run benchmark:smoke    # all models, 3 docs, q4+int8+q4f16 (Jina)
npm run leaderboard        # regenerate LEADERBOARD.md from results/
npm run benchmark          # full run (all models & quants, 54 docs)
```

Full runs download ONNX weights from Hugging Face on first use and cache them under `.cache/transformers-node/`.

## Repository layout

| Path | Purpose |
|------|---------|
| `data/benchmark-corpus.json` | 54 long documents (27 SV + 27 TR): mortgage, legal, medical |
| `config/models.mjs` | Embedding model registry with Hugging Face links |
| `config/gemma4-models.mjs` | Gemma 4 LLM registry (E2B/E4B, standard + QAT mobile) |
| `lib/transformers-runtime.mjs` | Node.js WASM / WebGPU / CPU runtime bootstrap |
| `scripts/benchmark.mjs` | Main benchmark runner |
| `scripts/generate-corpus.mjs` | Regenerates the corpus JSON |
| `results/` | Benchmark output JSON (gitignored) |

## WASM on Node.js

Transformers.js defaults to `onnxruntime-node` in Node. This project forces WASM by:

1. Injecting `onnxruntime-web` into `globalThis[Symbol.for('onnxruntime')]` **before** importing `@huggingface/transformers`.
2. Setting `env.useWasmCache = false` and preloading `wasmBinary` (Node cannot `fetch()` `file://` WASM URLs).
3. Passing `device: 'auto'` and `session_options: { executionProviders: ['wasm'] }` to `pipeline('feature-extraction', ...)`.
4. Mounting external `.onnx_data` shards via `session_options.externalData` (Transformers.js omits them in Node by default).

See `lib/transformers-runtime.mjs` for bootstrap details.

### WASM variants

| Bundle | ORT files | Notes |
|--------|-----------|-------|
| `wasm` (asyncify) | `ort-wasm-simd-threaded.asyncify.*` | Default; lacks WebGPU JSEP ops |
| `wasm-jsep` | `ort-wasm-simd-threaded.jsep.*` | **Required** for `GatherBlockQuantized` (EmbeddingGemma q4/q4f16) |

Auto backend order: **wasm-jsep → wasm → cpu**.

## Benchmarked models

All models are loaded from their Hugging Face `onnx/` folders:

| Model | Hugging Face ONNX folder |
|-------|--------------------------|
| Granite Embedding 278M Multilingual | https://huggingface.co/sirasagi62/granite-embedding-278m-multilingual-ONNX/tree/main/onnx |
| GTE Multilingual Base | https://huggingface.co/onnx-community/gte-multilingual-base/tree/main/onnx |
| Jina Embeddings v5 Omni Nano (**text**) | https://huggingface.co/onnx-community/jina-embeddings-v5-omni-nano-ONNX/tree/main/onnx |
| EmbeddingGemma 300M | https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/tree/main/onnx |
| Qwen3 Embedding 0.6B | https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/tree/main/onnx |
| BGE-M3 | https://huggingface.co/onnx-community/bge-m3-ONNX/tree/main/onnx |

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
  - `retrieval.recall_at_1`, `recall_at_3`, `recall_at_5`, `recall_at_10`
  - `composite_score` (weighted blend of cross-lingual, discrimination, recall@5)

Run-level summary includes wall time, peak RSS, leaderboard, and best variant per model.

## Runtime & crash resilience

Each variant runs in an **isolated subprocess** (`scripts/benchmark-variant.mjs`) so OOM kills do not crash the full benchmark.

**Backend strategy (`auto`):**
1. Try **wasm-jsep** (onnxruntime-web JSEP bundle — supports `GatherBlockQuantized`)
2. Fall back to **wasm** (asyncify bundle)
3. On failure (OOM, missing shards, unsupported ops), retry **CPU** (onnxruntime-node)

Models with known WASM issues use `backend: 'cpu'` directly (Jina text).

### EmbeddingGemma 300M

| Quant | Node WASM (jsep) | Node CPU | Browser WebGPU |
|-------|------------------|----------|----------------|
| q4 / q4f16 | **works** with `externalData` mount + jsep bundle | works | **q4 works**; q4f16 needs `shader-f16` GPU |
| no_gather_q4 | not needed (q4 only) | works | — |

**q4f16 probe results (Node 22, this repo):**

| Backend | Status | ms/doc (5 docs) | Notes |
|---------|--------|-----------------|-------|
| wasm-jsep | ok | ~2094 | `GatherBlockQuantized` via JSEP |
| wasm (asyncify) | fail | — | `GatherBlockQuantized` kernel missing |
| cpu | ok | ~22 | onnxruntime-node |
| webgpu (Node + Dawn) | fail | — | No Vulkan adapter on headless VM |
| webgpu (Chrome) q4f16 | partial | — | Loads ~5s; infer blocked without `shader-f16` |
| webgpu (Chrome) q4 | **ok** | ~12000 (1 doc) | Full inference on software Google adapter |

**WebGPU strategies tried (all failed for q4f16 except load):** default ANGLE, vulkan ANGLE, swiftshader, angle=gl, lavapipe ICD, Chrome blog flags (`--disable-vulkan-surface`), `forceCpuNodeNames` on Gather node, webgpu+wasm dual EP.

Probe scripts: `npm run probe:embeddinggemma`, `npm run probe:webgpu`.

Example full-corpus result (54 docs, CPU): quality **~0.63**, cross-lingual cosine **~0.81**, XL-R@5 **0.72**.

## Gemma 4 ONNX LLMs (text generation)

Four Hugging Face repos (text-only via `Gemma4ForCausalLM` — loads `embed_tokens` + `decoder_model_merged` only):

| Model | Hub ONNX folder | Quants |
|-------|-----------------|--------|
| Gemma 4 E2B IT | https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/tree/main/onnx | fp32, fp16, q4, q4f16, q8 (`_quantized`) |
| Gemma 4 E4B IT | https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX/tree/main/onnx | fp32, fp16, q4, q4f16, q8 |
| Gemma 4 E2B IT QAT Mobile | https://huggingface.co/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX/tree/main/onnx | **q2f16 only** |
| Gemma 4 E4B IT QAT Mobile | https://huggingface.co/onnx-community/gemma-4-E4B-it-qat-mobile-ONNX/tree/main/onnx | **q2f16 only** — [Getting Started](https://huggingface.co/onnx-community/gemma-4-E4B-it-qat-mobile-ONNX#getting-started) requires ORT 1.27+, WebGPU-first |

Probe matrix (model × quant × backend):

```bash
npm run build:ort                                       # build local ORT 1.28 node (see docs/gemma4-q2f16.md)
npm run build:ort:web                                   # build onnxruntime-web WASM (base, jsep, webgpu)
npm run build:ort:all                                   # node + web
npm run verify:ort:q2f16                                # smoke: 2-bit GatherBlockQuantized (CPU)
npm run verify:ort:web:q2f16                            # smoke: embed_tokens via wasm-jsep
npm run probe:gemma4:quick                              # E2B q4 smoke (4 backends)
npm run probe:gemma4                                    # full matrix (slow; multi-GB)
node scripts/probe-gemma4-matrix.mjs --model E2B-it --dtype q4,q8 --backend cpu,wasm-jsep
```

Backends tested: **cpu**, **wasm-jsep**, **wasm** (asyncify), **webgpu** (headless Chrome).

**WASM in Node:** pass `use_external_data_format: {}` and hub-relative `session_options.externalData` for all `embed_tokens*` + `decoder_model_merged*` shards (`lib/transformers-runtime.mjs` → `createTextGenerator`).

**Known limitations:**

| Issue | Symptom |
|-------|---------|
| q2f16 (mobile QAT) | `embed_tokens` uses `GatherBlockQuantized` **bits=2** — fails on ORT ≤1.26 CPU/WebGPU; decoder `MatMulNBits` 2-bit **loads on CPU** |
| q2f16 fix | **Local:** `npm run build:ort:all` → onnxruntime-node/web **1.28**; reinstall with `ONNXRUNTIME_NODE_INSTALL=skip npm install`; or wait for ORT **1.27+** npm |
| WASM asyncify | Missing `GatherBlockQuantized` for q4/q4f16 — use **wasm-jsep** |
| WASM infer OOM | E2B/E4B decoder is multi-GB; load may succeed but `std::bad_alloc` on infer |
| WebGPU q4f16/q2f16 | Needs `shader-f16`; probe falls back to q4 in browser |
| fp32 / E4B | Very large shard counts and RAM; expect slow downloads |

See **[docs/gemma4-q2f16.md](./docs/gemma4-q2f16.md)** for 2-bit gather research (ops anatomy, three ORT code paths, upgrade checklist).

Results: `results/probe-gemma4-matrix-<timestamp>.json` with per-cell `load_ms`, `infer_ms`, `status` (`ok` / `infer_error` / `load_error`).

**Benchmark** (multi-prompt timing, memory, tok/s):

```bash
npm run benchmark:gemma4:quick     # E2B q4+q8 cpu+webgpu, 3 prompts
npm run benchmark:gemma4           # all models × quants × backends (fp32 skipped by default)
npm run leaderboard:gemma4         # GEMMA4_LEADERBOARD.md
node scripts/benchmark-gemma4.mjs --model E2B-it,E4B-it --backend cpu --max-prompts 3
node scripts/probe-gemma4-hard.mjs # WebGPU strategy sweep (Chrome)
```

See **[GEMMA4_LEADERBOARD.md](./GEMMA4_LEADERBOARD.md)** for merged benchmark results.

## Retrieval metrics (robust)

| Metric | Meaning |
|--------|---------|
| **R@3** (`topic_any`) | Same topic in top 3, any language |
| **R@5** (`topic_any`) | Same topic in top 5, any language |
| **XL-R@5** (`topic_cross_lang`) | Same topic in top 5, **different** language (SV↔TR) — harder, more meaningful for this benchmark |
| **MRR@10** | Mean reciprocal rank of first relevant hit |
| **XLing** | Mean cosine of paired SV/TR documents on identical topics |

Composite score weights: cross-lingual cosine (35%), topic discrimination (25%), XL-R@5 (25%), R@5 (15%).

Record failures instead of hiding them:

| Issue | Typical symptom |
|-------|-----------------|
| External data (`*.onnx_data`) missing | fp32 init error for split ONNX graphs |
| External data not mounted (Node WASM) | Pass `session_options.externalData` with shard path/buffer |
| `q4f16` dtype mismatch | ORT graph fusion / FP16 cast issues (Granite, GTE, BGE-M3) |
| `GatherBlockQuantized` | Use **wasm-jsep** bundle, or WebGPU with `shader-f16`, or `no_gather_q4` variant |
| WebGPU `shader-f16` | q4f16 `GatherBlockQuantized` needs GPU f16 shaders; software adapters often lack it |

## Agent workflow

When extending this repo:

1. **Add an embedding model** — edit `config/models.mjs` with `id`, `name`, `url`, and any `model_file_name` / `extra_variants`.
2. **Add a Gemma 4 LLM** — edit `config/gemma4-models.mjs`; use `createTextGenerator` from `lib/transformers-runtime.mjs`.
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
