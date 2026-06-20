---
name: ort-128-e2b-q4
description: Build and wire ONNX Runtime 1.28 for Gemma 4 E2B-it q4 on Node (CPU/WASM) and browser (WebGPU/WASM). Use when setting up local ORT, onnxruntime-web bundles, package.json overrides, or running E2B q4 on server and web.
---

# ORT 1.28 + Gemma 4 E2B-it q4 (server & web)

End-to-end setup for **`onnx-community/gemma-4-E2B-it-ONNX`** with **`dtype: q4`** using a **local ONNX Runtime 1.28** build. Covers Node server (CPU, WASM-JSEP) and browser (WebGPU, WASM).

## When to use

- You need ORT **1.28** (not the bundled 1.24 node / 1.26-dev web in `@huggingface/transformers` 4.2.0).
- You want **E2B q4** on **CPU server** (~7% faster than bundled ORT 1.24 on this repo's benchmarks).
- You want **one ORT build** shared by **onnxruntime-node** and **onnxruntime-web** for server + browser parity.

E2B q4 also runs on bundled ORT for CPU; use this skill when you explicitly want 1.28 or plan to share the same ORT across web and server.

## Model & files

| Item | Value |
|------|-------|
| Hub repo | `onnx-community/gemma-4-E2B-it-ONNX` |
| Quant | `q4` → ONNX suffix `_q4` |
| Sessions | `embed_tokens_q4.onnx` + `decoder_model_merged_q4.onnx` (+ `.onnx_data` shards) |
| Transformers.js | `@huggingface/transformers` ^4.2.0 |

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Disk** | ~10 GB for ORT source + build artifacts (`vendor/onnxruntime/`, gitignored) |
| **RAM** | ≥4 GB for E2B q4 CPU inference (~2 GB peak RSS) |
| **Build tools** | `cmake`, `git`, `python3`, `gcc`/`g++` (not clang-only — linker needs libstdc++) |
| **Node** | 18+ with npm |
| **Web (optional)** | Chrome 121+ with WebGPU; emsdk pulled automatically by web build script |
| **Browser probes** | `playwright-core` + local Chrome (`devDependencies` in this repo) |

## 1. Build ORT 1.28 locally

From repo root (`onnx-lab`):

```bash
# Node bindings (~15–25 min on 4 cores)
npm run build:ort

# WASM + web JS bundles (~30–60 min additional; needs build:ort source)
npm run build:ort:web

# Or both:
npm run build:ort:all
```

Scripts:

- `scripts/build-ort.sh` — clones `vendor/onnxruntime` (shallow), builds Release + `--build_nodejs`, installs `js/{common,node}` deps.
- `scripts/build-ort-web.sh` — builds three WASM variants (base, jsep, webgpu/asyncify), copies to `vendor/onnxruntime/js/web/dist`, runs `npm run build` for bundles.

Verify build:

```bash
cat vendor/onnxruntime/VERSION_NUMBER          # expect 1.28.0
ls vendor/onnxruntime/js/node/bin/napi-v6/linux/x64/onnxruntime_binding.node
ls vendor/onnxruntime/js/web/dist/ort.min.mjs
```

**Compiler pitfall:** if `c++` is clang without `-lstdc++`, force gcc:

```bash
export CC=gcc CXX=g++
npm run build:ort:all
```

## 2. Wire npm to local ORT

Add to `package.json`:

```json
{
  "dependencies": {
    "@huggingface/transformers": "^4.2.0",
    "onnxruntime-common": "file:vendor/onnxruntime/js/common",
    "onnxruntime-node": "file:vendor/onnxruntime/js/node",
    "onnxruntime-web": "file:vendor/onnxruntime/js/web"
  },
  "overrides": {
    "@huggingface/transformers": {
      "onnxruntime-common": "file:vendor/onnxruntime/js/common",
      "onnxruntime-node": "file:vendor/onnxruntime/js/node",
      "onnxruntime-web": "file:vendor/onnxruntime/js/web"
    },
    "onnxruntime-common": "file:vendor/onnxruntime/js/common",
    "onnxruntime-node": "file:vendor/onnxruntime/js/node",
    "onnxruntime-web": "file:vendor/onnxruntime/js/web"
  }
}
```

Install (skip prebuilt node binary download — use your local binding):

```bash
ONNXRUNTIME_NODE_INSTALL=skip npm install
```

Verify versions:

```bash
node -e "import('onnxruntime-node').then(m => console.log('node ORT', m.env.versions?.common))"
node -e "import('onnxruntime-web/package.json', {with:{type:'json'}}).then(m => console.log('web ORT', m.default.version))"
```

Both should report **1.28.0**.

## 3. Server — Node CPU (recommended)

Best throughput and stability for E2B q4 on server. Uses `onnxruntime-node` automatically via Transformers.js when `device: 'cpu'`.

```javascript
import { createTextGenerator } from './lib/transformers-runtime.mjs';

const MODEL = 'onnx-community/gemma-4-E2B-it-ONNX';

const generator = await createTextGenerator(
  MODEL,
  { dtype: 'q4' },
  'cpu',
);

const messages = [{ role: 'user', content: 'Explain a Swedish bolån in one sentence.' }];
const out = await generator(messages, {
  max_new_tokens: 64,
  do_sample: false,
  return_full_text: false,
});
console.log(out[0].generated_text);
```

**Smoke test (one prompt):**

```bash
node --expose-gc scripts/benchmark-gemma4-variant.mjs \
  --model-id onnx-community/gemma-4-E2B-it-ONNX \
  --model-slug E2B-it \
  --model-name "Gemma 4 E2B IT" \
  --dtype q4 \
  --backend cpu \
  --max-prompts 1
```

**Benchmark (6 prompts):**

```bash
node --expose-gc scripts/benchmark-gemma4.mjs \
  --model E2B-it --dtype q4 --backend cpu
```

First run downloads ONNX shards to `.cache/transformers-node/` (~1–2 GB).

## 4. Server — Node WASM-JSEP (browser-aligned ONNX path)

Use when you need the same WASM execution path as a browser app, without native `onnxruntime-node`. Requires **external data** mounting for Gemma 4 shards — handled by `createTextGenerator` in `lib/transformers-runtime.mjs`.

```javascript
import { createTextGenerator } from './lib/transformers-runtime.mjs';

const generator = await createTextGenerator(
  'onnx-community/gemma-4-E2B-it-ONNX',
  { dtype: 'q4' },
  'wasm-jsep',   // NOT 'wasm' — asyncify lacks GatherBlockQuantized for q4
);
```

**Important limitations for E2B q4:**

| Backend | Load | Infer |
|---------|------|-------|
| `wasm` (asyncify) | ❌ `gather_quant` | — |
| `wasm-jsep` | ✅ | ⚠️ often OOM on VMs with <8 GB RAM |

Probe:

```bash
npm run probe:gemma4:quick   # E2B q4 × cpu, wasm-jsep, wasm, webgpu
```

## 5. Web — browser WebGPU (recommended for in-browser)

E2B **q4** does **not** require `shader-f16` (unlike q4f16/q2f16). WebGPU is the intended fast path.

### 5a. Bundled app (Vite / Next / etc.) with local ORT 1.28

After step 2 (`onnxruntime-web` from `file:vendor/...`):

```javascript
import { pipeline, env } from '@huggingface/transformers';

env.allowRemoteModels = true;
env.useBrowserCache = true;

// Optional: serve WASM from your CDN/public folder
// Copy vendor/onnxruntime/js/web/dist/ort-wasm-simd-threaded.jsep.* to /public/ort/
// env.backends.onnx.wasm.wasmPaths = '/ort/';

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
const device = await adapter.requestDevice();
env.backends.onnx.webgpu = env.backends.onnx.webgpu ?? {};
env.backends.onnx.webgpu.device = device;

const generator = await pipeline('text-generation', 'onnx-community/gemma-4-E2B-it-ONNX', {
  dtype: 'q4',
  device: 'webgpu',
  session_options: { enableMemPattern: false },
});

const out = await generator('Hello, my name is', { max_new_tokens: 32, do_sample: false });
```

**Bundler notes:**

- Resolve `onnxruntime-web` from your local `file:` package (overrides ensure Transformers.js uses 1.28).
- For WASM backends, copy these from `node_modules/onnxruntime-web/dist/` to your static assets:
  - `ort-wasm-simd-threaded.jsep.mjs` + `.wasm` (WASM-JSEP)
  - `ort-wasm-simd-threaded.asyncify.mjs` + `.wasm` (WebGPU fallback path)
- Set `env.backends.onnx.wasm.wasmPaths` if files are not at the default CDN path.

### 5b. Headless Chrome probe (this repo)

Uses Playwright + a minimal HTML page (`lib/gemma4-helpers.mjs` → `gemma4BrowserHtml`). **Note:** the probe HTML imports Transformers from **jsDelivr CDN** (bundled ORT, not your local 1.28). Use this to validate WebGPU + q4 behavior; for local ORT 1.28 in browser, use 5a.

```bash
node scripts/probe-gemma4-webgpu-strategy.mjs \
  onnx-community/gemma-4-E2B-it-ONNX \
  browser:q4-control
```

Strategies: `browser:q4-control`, `browser:q4-force-gather`, `browser:q4-dual-ep` (see `lib/gemma4-helpers.mjs`).

### 5c. Web — WASM in browser

Same as server WASM-JSEP but in browser. Set wasm paths, use `device: 'auto'` with `executionProviders: ['wasm']`. E2B q4 is large — expect slow load and high memory; WebGPU q4 is preferred.

## 6. Verification checklist

Run in order after setup:

```bash
# 1. ORT versions
node -e "import('onnxruntime-node').then(m => console.log(m.env.versions?.common))"

# 2. CPU text-gen smoke
node --expose-gc scripts/benchmark-gemma4-variant.mjs \
  --model-id onnx-community/gemma-4-E2B-it-ONNX \
  --model-slug E2B-it --model-name "Gemma 4 E2B IT" \
  --dtype q4 --backend cpu --max-prompts 1

# 3. WASM-JSEP load smoke (may OOM on infer — load success is the gate)
node --expose-gc scripts/probe-gemma4-matrix.mjs \
  --model E2B-it --dtype q4 --backend wasm-jsep

# 4. WebGPU browser smoke (CDN transformers; q4 control strategy)
node scripts/probe-gemma4-webgpu-strategy.mjs \
  onnx-community/gemma-4-E2B-it-ONNX browser:q4-control
```

Expected for E2B q4:

| Path | Expected |
|------|----------|
| CPU | `status: ok`, ~8–9 tok/s, RSS ~1.9 GB |
| wasm-jsep | load ok; infer may OOM on small VMs |
| wasm (asyncify) | `gather_quant` error — use wasm-jsep |
| WebGPU q4 | ok on real GPU Chrome; headless may need `--enable-unsafe-webgpu` |

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `bits==4 or 8` / `gather_quant` | ORT too old or wrong WASM variant | Use local ORT 1.28; backend `wasm-jsep` not `wasm` |
| `onnxruntime_binding.node` missing | Skipped build or wrong platform | Re-run `npm run build:ort`; check `napi-v6/linux/x64/` path |
| `external data` / `onnx_data` | Shards not mounted (WASM) | Use `createTextGenerator` — it sets `gemma4WasmExternalData()` |
| `std::bad_alloc` / OOM | E2B decoder too large for WASM RAM | Use **CPU** or **WebGPU**; need ≥8 GB for wasm-jsep infer |
| `shader-f16` / `requires f16` | q4f16 or q2f16 on WebGPU without f16 | Use **`q4`** for E2B; or fallback to q4 |
| C++ link errors during build | clang default | `export CC=gcc CXX=g++` before build |
| `npm install` downloads wrong ORT | Missing overrides | Add `overrides` for `@huggingface/transformers` + direct deps |
| Browser still on ORT 1.26 | CDN import | Bundle local `@huggingface/transformers` + `onnxruntime-web` (section 5a) |

## 8. Architecture (quick reference)

```
@huggingface/transformers 4.2.0
├── Server CPU     → onnxruntime-node 1.28 (file:vendor/.../js/node)
├── Server WASM    → onnxruntime-web 1.28 + jsep wasm (bootstrapOrt in lib/transformers-runtime.mjs)
└── Browser WebGPU → onnxruntime-web/webgpu 1.28 (bundler) or CDN (probes only)

Gemma 4 E2B q4 sessions:
  embed_tokens_q4.onnx + decoder_model_merged_q4.onnx (+ external .onnx_data)
```

## 9. Related docs in this repo

| File | Contents |
|------|----------|
| `lib/transformers-runtime.mjs` | `createTextGenerator`, `bootstrapOrt`, WASM external data |
| `config/gemma4-models.mjs` | Model IDs, quant suffixes |
| `scripts/build-ort.sh` / `build-ort-web.sh` | Build scripts |
| `GEMMA4_ORT128.md` | ORT 1.28 vs 1.24 speed comparison (E2B q4 +7% tok/s) |
| `docs/gemma4-q2f16.md` | Deep ORT build notes (q2f16; same build pipeline) |
| `AGENTS.md` | Gemma 4 matrix commands |

## 10. When ORT 1.28 ships on npm

Replace `file:` deps with published versions and remove overrides:

```json
{
  "dependencies": {
    "@huggingface/transformers": "^4.x",
    "onnxruntime-node": "^1.28.0",
    "onnxruntime-web": "^1.28.0"
  }
}
```

Re-run the verification checklist. Drop `ONNXRUNTIME_NODE_INSTALL=skip` unless you still build from source.
