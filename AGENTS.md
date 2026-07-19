# onnx-lab

ONNX benchmark lab for multilingual embeddings and Gemma 4 with
Transformers.js on CPU, Node WASM, and browser WebGPU.

## Start here

```bash
npm install
npm run check
npm test
```

Use targeted smoke runs while developing:

```bash
node scripts/benchmark.mjs \
  --model onnx-community/embeddinggemma-300m-ONNX \
  --dtype q4 --max-texts 1

node scripts/benchmark-gemma4.mjs \
  --model E2B-it --dtype q4 --backend cpu --max-prompts 1

node scripts/eval-gemma4-quality.mjs \
  --variant E2B-it:q4 --category mcq --max-tasks 1

node scripts/eval-gemma4-multimodal.mjs \
  --model E2B-it --dtype q4 --modality image --max-tasks 1
```

Full matrices download several gigabytes and can take hours.

## Essential modules

| Path | Responsibility |
|------|----------------|
| `config/models.mjs` | Embedding registry, dtype aliases/suffixes, backend selection |
| `config/gemma4-models.mjs` | Gemma 4 registry, supported quants/backends |
| `lib/transformers-runtime.mjs` | CPU/WASM/JSEP pipeline initialization and external data |
| `lib/benchmark-support.mjs` | Root paths, JSON, isolated workers, timing, memory, error classification |
| `lib/browser-runtime.mjs` | Ephemeral HTTP server plus Chrome execution |
| `lib/gemma4-webgpu.mjs` | Gemma 4 browser strategies and generated probe page |
| `lib/metrics.mjs` | Embedding quality and retrieval metrics |
| `lib/gemma4-quality-scoring.mjs` | Deterministic quality evaluation |
| `lib/gemma4-multimodal-runtime.mjs` | CPU image/audio loading and generation |
| `data/*.json` | Committed benchmark and evaluation inputs |

## Data flow

Embedding benchmarks:

```text
scripts/benchmark.mjs
  -> one scripts/benchmark-variant.mjs process per model/quant/backend
  -> createFeatureExtractor()
  -> computeQuality()
  -> results/benchmark-*.json
  -> scripts/generate-leaderboard.mjs
```

Gemma 4 benchmarks and evaluations:

```text
matrix/eval parent
  -> isolated variant/worker process
  -> createTextGenerator() or multimodal runtime
  -> CPU/WASM inference or generated Chrome WebGPU page
  -> results/*.json
```

Workers write JSON through `runJsonWorker()`. Keep model loads isolated; the
process boundary is intentional OOM protection.

## Runtime initialization

- CPU removes any injected web ORT and lets Transformers.js use
  `onnxruntime-node`.
- `wasm` and `wasm-jsep` inject `onnxruntime-web` before dynamically importing
  Transformers.js.
- Node cannot fetch local `file://` WASM, so `bootstrapOrt()` preloads the WASM
  binary.
- `wasm-jsep` is required for `GatherBlockQuantized`.
- Gemma 4 WASM sessions mount all configured `embed_tokens` and
  `decoder_model_merged` external data shards.
- Browser WebGPU runs through `lib/browser-runtime.mjs`; do not add fixed ports
  or duplicate Playwright launch code.

## Change guidance

- Add embedding models in `config/models.mjs`.
- Add Gemma 4 models in `config/gemma4-models.mjs`.
- Put shared runner behavior in `lib/benchmark-support.mjs`, not individual
  scripts.
- Keep browser strategy logic in `lib/gemma4-webgpu.mjs` or the maintained
  EmbeddingGemma browser probe.
- Treat `data/gemma4-quality-suite.json` and
  `data/gemma4-multimodal-suite.json` as source data.
- `scripts/generate-corpus.mjs` must remain deterministic; run it and confirm no
  diff after corpus changes.
- Keep result schemas backward-compatible where practical because leaderboard
  generators merge historical result files.
- Record model/backend failures in output JSON instead of suppressing them.
- Do not commit `.cache/`, `results/`, `node_modules/`, or `vendor/onnxruntime/`.

## Verification

Run after code changes:

```bash
npm run check
npm test
npm run generate:corpus
git diff --exit-code -- data/benchmark-corpus.json
```

Then run the narrowest real-model command covering the changed path. Use full
benchmarks only before publishing benchmark results.

Optional ORT source builds remain under `scripts/build-ort*.sh`; normal
development uses the pinned npm ORT 1.27.0 packages.
