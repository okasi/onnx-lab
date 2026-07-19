# onnx-lab

ONNX model experiments for multilingual embeddings and Gemma 4 generation with
Transformers.js across CPU, Node WASM, and browser WebGPU.

## Setup

```bash
npm install
npm run check
npm test
```

Model weights are downloaded on first use and cached under
`.cache/transformers-node/`. Benchmark output is written to `results/`; both
directories are gitignored.

## Main commands

```bash
# Embeddings
npm run benchmark:quick
npm run benchmark
npm run leaderboard
npm run probe:embeddinggemma
npm run probe:gemma-quants
npm run probe:webgpu

# Gemma 4 text generation
npm run probe:gemma4:quick
npm run benchmark:gemma4:quick
npm run leaderboard:gemma4
npm run eval:gemma4:quality

# Gemma 4 image/audio
npm run eval:gemma4:multimodal

# Data and runtime verification
npm run generate:corpus
npm run verify:ort:q2f16
npm run verify:ort:web:q2f16
```

Every expensive evaluator supports a small targeted run. Examples:

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

## Architecture

- `config/` contains the embedding and Gemma 4 model registries.
- `lib/transformers-runtime.mjs` creates CPU, WASM, and JSEP pipelines.
- `lib/benchmark-support.mjs` owns paths, JSON I/O, subprocess isolation, timing, and memory measurement.
- `lib/browser-runtime.mjs` runs generated probe pages in Chrome on an ephemeral local port.
- `lib/metrics.mjs` computes embedding retrieval and cross-lingual quality.
- `lib/gemma4-quality-scoring.mjs` contains deterministic quality scorers.
- `scripts/*-variant.mjs` and `scripts/*-worker.mjs` isolate model loads so OOM failures do not kill matrix runs.
- `data/` contains the committed benchmark corpus and evaluation suites.

Embedding flow:

```text
benchmark.mjs -> benchmark-variant.mjs -> transformers-runtime
              -> metrics -> results/*.json -> generate-leaderboard.mjs
```

Gemma 4 flow:

```text
matrix/eval command -> isolated worker -> CPU/WASM pipeline or Chrome WebGPU page
                    -> results/*.json -> report generator
```

## Runtime notes

The package lock pins ONNX Runtime 1.27.0 and overrides the versions requested by
Transformers.js so Node and web runtimes use the same ORT release.

Node WASM is initialized before Transformers.js is imported. `wasm-jsep` is
required for quantized gather operators used by EmbeddingGemma and Gemma 4.
Gemma 4 WASM sessions also mount external ONNX data shards from the model cache.

WebGPU runs in the installed Chrome channel. Mixed-precision variants
may require the `shader-f16` adapter feature; q4 is the practical fallback when
that feature is unavailable.

See [AGENTS.md](./AGENTS.md) for contribution guidance,
[LEADERBOARD.md](./LEADERBOARD.md) for embedding results, and
[GEMMA4_LEADERBOARD.md](./GEMMA4_LEADERBOARD.md) for text-generation results.
