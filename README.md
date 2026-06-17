# embedding-tests

Swedish and Turkish embedding benchmarks for ONNX multilingual models using **Transformers.js WASM on Node.js**.

## What's inside

- **54 long documents** (Swedish + Turkish) on mortgages, legal, and medical topics
- Benchmark runner for **9 Hugging Face ONNX embedding models** and their quantization variants
- WASM runtime bootstrap for Node.js (`lib/transformers-wasm.mjs`)

See **[AGENTS.md](./AGENTS.md)** for full instructions, model links, CLI flags, and agent workflow.

See **[LEADERBOARD.md](./LEADERBOARD.md)** for the full ranked results table (all models, quants, quality, memory, and timing).

## Commands

```bash
npm install
npm run benchmark:quick   # fast smoke test
npm run benchmark         # full matrix (slow; downloads models)
```

Results are written to `results/benchmark-<timestamp>.json`.
