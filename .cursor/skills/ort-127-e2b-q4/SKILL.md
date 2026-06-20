---
name: ort-127-e2b-q4
description: Install ONNX Runtime 1.27.0 from npm for Gemma 4 E2B-it q4 on Node (CPU/WASM) and browser (WebGPU/WASM). Use when setting up ORT, package.json overrides, or running E2B q4 on server and web.
---

# ORT 1.27 + Gemma 4 E2B-it q4 (server & web)

Use **npm `onnxruntime-*@1.27.0`** — no local ORT build required.

**Canonical install guide:** [docs/ort-127-install.md](../../docs/ort-127-install.md)

## Quick install

```bash
npm install
node -e "import('onnxruntime-node').then(m => console.log('ORT', m.env.versions?.common))"
# → 1.27.0
```

This repo's `package.json` already pins 1.27.0 and overrides `@huggingface/transformers`.

## Model

| Item | Value |
|------|-------|
| Hub | `onnx-community/gemma-4-E2B-it-ONNX` |
| Quant | `q4` (`embed_tokens_q4.onnx` + `decoder_model_merged_q4.onnx`) |

## Server — CPU (recommended)

```javascript
import { createTextGenerator } from './lib/transformers-runtime.mjs';

const gen = await createTextGenerator(
  'onnx-community/gemma-4-E2B-it-ONNX',
  { dtype: 'q4' },
  'cpu',
);
```

Smoke test:

```bash
node --expose-gc scripts/benchmark-gemma4-variant.mjs \
  --model-id onnx-community/gemma-4-E2B-it-ONNX \
  --model-slug E2B-it --model-name "Gemma 4 E2B IT" \
  --dtype q4 --backend cpu --max-prompts 1
```

## Server — WASM-JSEP

Use `wasm-jsep` (not `wasm`). External data is handled in `lib/transformers-runtime.mjs`.

## Browser — WebGPU

```javascript
import { pipeline, env } from '@huggingface/transformers';

const device = await (await navigator.gpu.requestAdapter()).requestDevice();
env.backends.onnx.webgpu = { device };

await pipeline('text-generation', 'onnx-community/gemma-4-E2B-it-ONNX', {
  dtype: 'q4',
  device: 'webgpu',
});
```

Headless probe: `node scripts/probe-gemma4-webgpu-strategy.mjs onnx-community/gemma-4-E2B-it-ONNX browser:q4-control`

## package.json template (other projects)

```json
{
  "dependencies": {
    "@huggingface/transformers": "^4.2.0",
    "onnxruntime-common": "1.27.0",
    "onnxruntime-node": "1.27.0",
    "onnxruntime-web": "1.27.0"
  },
  "overrides": {
    "@huggingface/transformers": {
      "onnxruntime-common": "1.27.0",
      "onnxruntime-node": "1.27.0",
      "onnxruntime-web": "1.27.0"
    }
  }
}
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Still on ORT 1.24 | Add `overrides`; reinstall `node_modules` |
| `gather_quant` on WASM | Use `wasm-jsep` backend |
| q2f16 fails | Confirm ORT **1.27.0** (`npm run verify:ort:q2f16`) |

## Optional: build ORT from source

Only for unreleased ORT `main`. See [docs/gemma4-q2f16.md](../../docs/gemma4-q2f16.md) and `npm run build:ort:all`.

## Related

- [docs/ort-127-install.md](../../docs/ort-127-install.md) — full install + verify steps
- [GEMMA4_ORT128.md](../../GEMMA4_ORT128.md) — speed benchmarks vs 1.24
- `lib/transformers-runtime.mjs` — `createTextGenerator`, `bootstrapOrt`
