# Gemma 4 mobile q2f16 notes

Current as of July 19, 2026.

The Gemma 4 mobile QAT ONNX repositories expose `q2f16` text graphs:

- `onnx-community/gemma-4-E2B-it-qat-mobile-ONNX`
- `onnx-community/gemma-4-E4B-it-qat-mobile-ONNX`

Both require ONNX Runtime 1.27 or newer for 2-bit
`GatherBlockQuantized`. This repository pins ORT 1.27.0.

## Graph layout

Text generation uses two ONNX sessions:

| Session | Important outputs/operations |
|---------|------------------------------|
| `embed_tokens_q2f16.onnx` | `inputs_embeds`, `per_layer_inputs`; 2-bit `GatherBlockQuantized` |
| `decoder_model_merged_q2f16.onnx` | KV cache and 2/4/8-bit `MatMulNBits` |

Gemma 4 uses per-layer embeddings, so `per_layer_inputs` from the embedding
session must reach the decoder.

The multimodal repositories also contain vision and audio encoder sessions.
Those are loaded by `Gemma4ForConditionalGeneration` in
`lib/gemma4-multimodal-runtime.mjs`.

## Backend status

| Backend | q2f16 status |
|---------|--------------|
| CPU (`onnxruntime-node` 1.27) | Supported; high memory and slower than q4 |
| Node `wasm-jsep` | Session load can work with external data; full decoder inference may OOM |
| Node `wasm` asyncify | Not suitable for quantized gather graphs |
| Browser WebGPU | Intended deployment path; requires an adapter with `shader-f16` |

The repository mounts configured external data shards for both text sessions in
`gemma4WasmExternalData()`.

## Verification

The low-level checks require a cached
`embed_tokens_q2f16.onnx`:

```bash
npm run verify:ort:q2f16
npm run verify:ort:web:q2f16
```

Targeted text generation:

```bash
node scripts/benchmark-gemma4.mjs \
  --model E2B-qat-mobile \
  --dtype q2f16 \
  --backend cpu \
  --max-prompts 1
```

WASM load/inference probe:

```bash
node scripts/probe-gemma4-matrix.mjs \
  --model E2B-qat-mobile \
  --dtype q2f16 \
  --backend cpu,wasm-jsep
```

## Failure interpretation

- `gather_quant`: runtime does not support the graph's quantized gather.
- `external_data`: an ONNX data shard is absent or not mounted.
- `oom`: the graph loaded but the process ran out of memory during inference.
- `shader_f16`: the WebGPU adapter cannot execute mixed-precision shaders.

Use standard Gemma 4 `q4` when q2f16 is not required. It is smaller operational
risk across CPU, WASM-JSEP, and WebGPU environments.
