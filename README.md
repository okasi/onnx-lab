# embedding-tests

Swedish and Turkish embedding benchmarks for ONNX multilingual models using **Transformers.js** on Node.js (WASM / CPU).

## Recommended pick: EmbeddingGemma 300M **q4f16**

After benchmarking six multilingual ONNX embedding models across every practical quant, **EmbeddingGemma 300M q4f16** is our overall winner — not because it tops the raw quality leaderboard (GTE and BGE-M3 score slightly higher), but because it wins on the combination that actually matters when you ship embeddings in production.

| Why it wins | EmbeddingGemma 300M q4f16 | Typical top scorer (e.g. GTE bnb4) |
|-------------|---------------------------|-------------------------------------|
| **Runs at all** | ✅ Stable on CPU; also works on **wasm-jsep** in Node | ✅ WASM, but… |
| **q4f16 support** | ✅ Only model in the suite where **q4f16 loads and runs end-to-end** | ❌ GTE / Granite: FP16 graph fusion crash; BGE-M3: FP16 tensor mismatch |
| **Memory** | **~585 MB** peak RSS | ~2–6.6 GB |
| **Speed** | **~92 ms/doc**, **6 s** total (54 docs) | ~928–2945 ms/doc, 55 s–3 min |
| **Quality** | **0.636** composite · XLing **0.816** · XL-R@5 **0.72** | 0.730 composite · XLing 0.953 · XL-R@5 0.96 |
| **Retrieval** | R@5 **1.00** · R@3 **0.96** · R@1 **0.91** | Comparable on R@5; much heavier to get there |
| **Model size** | ~168 MB weights (external `.onnx_data`) | Multi-GB working set at runtime |
| **Dim** | 768 — compact vectors | 768–1024 |

### What that means in practice

1. **q4f16 that actually works.** We tested `q4f16` on every model. Four variants failed outright (Granite, GTE, BGE-M3, Jina). Qwen3 q4f16 runs but is far slower and weaker on cross-lingual retrieval. EmbeddingGemma q4f16 is the **only q4f16 quant that is fast, small, and reliable** in this benchmark.

2. **Near-identical quality to other EmbeddingGemma quants.** q4f16 (0.6362) matches q4 (0.6361) and no_gather_q4 (0.6363) on composite score and retrieval — you do not sacrifice accuracy by choosing the mixed-precision quant.

3. **Best speed–memory–quality trade-off in the Gemma family.** q4 is ~12% faster (80 ms/doc) but q4f16 still finishes the full 54-document corpus in **6 seconds** at **under 600 MB** RSS — orders of magnitude leaner than WASM runs of GTE/BGE that need 2–6 GB.

4. **Strong Swedish/Turkish cross-lingual behavior.** XLing **0.816** and XL-R@5 **0.72** on paired SV↔TR documents are solid for a 300M model; same-topic R@5 hits **1.00**.

5. **Deployable beyond CPU.** With the hardened runtime in this repo (JSEP WASM bundle + external data mount), q4f16 also runs on **wasm-jsep** in Node — important for browser-aligned ONNX paths and for environments without native ORT.

6. **WebGPU-ready when hardware allows.** In Chrome, the q4f16 graph loads on WebGPU; inference needs a GPU with `shader-f16` (discrete GPUs). Software renderers load the model but cannot execute the q4f16 gather shader — a hardware constraint, not a model defect.

**Hub:** [onnx-community/embeddinggemma-300m-ONNX](https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/tree/main/onnx) · ONNX file: `model_q4f16.onnx` + `model_q4f16.onnx_data`

Full numbers: **[LEADERBOARD.md](./LEADERBOARD.md)** (rank 12 by composite score; rank 1 by deployable q4f16 efficiency).

---

## What's inside

- **54 long documents** (Swedish + Turkish) on mortgages, legal, and medical topics
- Benchmark runner for **6 Hugging Face ONNX embedding models** and their quantization variants
- Runtime bootstrap for Node.js WASM / CPU / WebGPU probes (`lib/transformers-runtime.mjs`)

See **[AGENTS.md](./AGENTS.md)** for full instructions, model links, CLI flags, and agent workflow.

## Commands

```bash
npm install
npm run benchmark:quick      # fast smoke test
npm run benchmark            # full matrix (slow; downloads models)
npm run probe:embeddinggemma # Node backend matrix for Gemma q4f16
npm run probe:webgpu         # Chrome WebGPU probe
npm run leaderboard          # regenerate LEADERBOARD.md from results/
```

Results are written to `results/benchmark-<timestamp>.json`.
