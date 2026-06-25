# Gemma 4 ONNX LLM Leaderboard

Merged from 3 benchmark run(s) (14 unique variants).

## Results

| Model | Quant | Backend | Status | Load (ms) | ms/prompt | tok/s | RSS (MB) | Notes |
|-------|-------|---------|--------|-----------|-----------|-------|----------|-------|
| E2B-it | fp16 | wasm | error | - | - | - | 191.57 | error |
| E2B-it | fp16 | wasm-jsep | error | - | - | - | 178.09 | error |
| E2B-it | q4 | cpu | ok | 3385.07 | 1465.17 | 5.46 | 1858.3 | - |
| E2B-it | q4 | wasm | error | - | - | - | 9018.06 | gather_quant |
| E2B-it | q4 | wasm-jsep | infer_error | 8730.3 | - | - | 10755.7 | oom; load-only |
| E2B-it | q4f16 | cpu | ok | 22554.29 | 1121.36 | 7.13 | 3773.58 | - |
| E2B-it | q4f16 | wasm | error | - | - | - | 8113.43 | gather_quant |
| E2B-it | q4f16 | wasm-jsep | infer_error | 11386.31 | - | - | 9431.91 | oom; load-only |
| E2B-it | q8 | cpu | ok | 56450.52 | 19189.22 | 0.42 | 5184.34 | - |
| E2B-it | q8 | wasm | error | - | - | - | 174.98 | error |
| E2B-it | q8 | wasm-jsep | error | - | - | - | 178.05 | error |
| E2B-qat-mobile | q2f16 | cpu | error | - | - | - | 180.64 | gather_quant |
| E4B-it | q4 | cpu | ok | 43970.12 | 1263.77 | 3.17 | 5127.15 | - |
| E4B-qat-mobile | q2f16 | cpu | error | - | - | - | 719.24 | gather_quant |

## Fastest CPU / working backends (mean ms/prompt)

- **E2B-it q4f16 cpu** — 1121.36 ms/prompt, 7.13 tok/s, RSS 3773.58 MB
- **E4B-it q4 cpu** — 1263.77 ms/prompt, 3.17 tok/s, RSS 5127.15 MB
- **E2B-it q4 cpu** — 1465.17 ms/prompt, 5.46 tok/s, RSS 1858.3 MB
- **E2B-it q8 cpu** — 19189.22 ms/prompt, 0.42 tok/s, RSS 5184.34 MB

## Backend compatibility (this environment)

| Backend | E2B q4 | E2B q4f16 | E4B q4 | Mobile q2f16 |
|---------|--------|-----------|--------|---------------|
| **cpu** | ok | ok | ok | gather_quant |
| **wasm-jsep** | load only | load only | — | — |
| **wasm** | gather_quant | — | — | — |
| **webgpu** | — | — | — | — |

## Failed / partial variants

- E2B-it fp16 wasm: **error** (error)
- E2B-it fp16 wasm-jsep: **error** (error)
- E2B-it q4 wasm: **error** (gather_quant)
- E2B-it q4 wasm-jsep: **infer_error** (oom) — loaded in 8730ms
- E2B-it q4f16 wasm: **error** (gather_quant)
- E2B-it q4f16 wasm-jsep: **infer_error** (oom) — loaded in 11386ms
- E2B-it q8 wasm: **error** (error)
- E2B-it q8 wasm-jsep: **error** (error)
- E2B-qat-mobile q2f16 cpu: **error** (gather_quant)
- E4B-qat-mobile q2f16 cpu: **error** (gather_quant)

## Source runs

- `benchmark-gemma4-1781833188722.json` (2026-06-19T01:39:48.715Z, 12 variants)
- `benchmark-gemma4-1781833890459.json` (2026-06-19T01:51:30.458Z, 2 variants)
- `benchmark-gemma4-1781833948083.json` (2026-06-19T01:52:28.076Z, 1 variants)
