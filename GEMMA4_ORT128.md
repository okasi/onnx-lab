# Gemma 4 — ORT version comparison (CPU speed)

Compared **bundled 1.24.3** (default in `@huggingface/transformers` 4.2.0), **npm 1.27.0**, and **local 1.28.0** (ORT `main` build) on CPU for the four production quants.

**Setup:** 6 prompts × 8 new tokens, greedy, `device: cpu`, models `onnx-community/gemma-4-{E2B,E4B}-it-ONNX`.

**Runs:** 3 passes for 1.24 and 1.28; **5 passes** for npm 1.27.0.

## Summary — average tok/s vs bundled 1.24

| Variant | ORT 1.24 | ORT 1.27 (npm, 5 runs) | ORT 1.28 (local, 3 runs) |
|---------|----------|------------------------|--------------------------|
| **E2B q4** | 8.49 ± 0.40 | **9.35 ± 0.27 (+10%)** | 9.11 ± 0.49 (+7%) |
| E2B q4f16 | 8.79 ± 0.18 | 8.84 ± 0.34 (+0.6%) | 8.62 ± 0.62 (−2%) |
| E4B q4 | 4.43 ± 0.23 | 4.59 ± 0.14 (+3.7%) | 4.36 ± 0.31 (−2%) |
| E4B q4f16 | 4.39 ± 0.28 | 4.50 ± 0.14 (+2.5%) | 4.38 ± 0.26 (−0%) |

Peak **RSS unchanged** (~1.9 / 2.3 / 3.4 / 4.0 GB).

## E2B q4 — per-run (best signal)

| Run | 1.24 tok/s | 1.27 tok/s | 1.28 tok/s |
|-----|------------|------------|------------|
| 1 | 8.08 | 8.88 | 8.99 |
| 2 | 8.57 | 9.22 | 8.57 |
| 3 | 9.03 | 9.53 | 9.76 |
| 4 | — | 9.64 | — |
| 5 | — | 9.47 | — |

All five 1.27 E2B-q4 runs beat the 1.24 mean (8.49). Lowest 1.27 run (8.88) is still above two of three 1.24 runs.

## Takeaways

- **npm `onnxruntime-node@1.27.0` gives a real speedup** — no local build required.
- **E2B q4** benefits most: **~10% tok/s** on 1.27 (5-run avg), ~7% on local 1.28.
- **E4B q4 / q4f16** show modest ~2–4% gains on 1.27; flat on local 1.28 (within noise).
- **1.27 npm slightly beats local 1.28** on this VM for E2B q4 — published binaries may be better optimized than a generic source build.
- **Practical recommendation:** override to **`1.27.0` from npm** unless you need ORT `main` features (e.g. unreleased ops).

## Install npm 1.27.0 (no build)

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

```bash
npm install
node -e "import('onnxruntime-node').then(m => console.log(m.env.versions?.common))"  # 1.27.0
```

## Raw results

| ORT | Runs | Files |
|-----|------|-------|
| 1.24 | 3 | `results/benchmark-gemma4-speed-4variants.json`, `…-ort124-run{2,3}.json` |
| 1.27 | 5 | `results/benchmark-gemma4-speed-ort127-run{1..5}.json` |
| 1.28 | 3 | `results/benchmark-gemma4-speed-ort128.json`, `…-ort128-run{2,3}.json` |

Local 1.28 build: `npm run build:ort:all` + `file:vendor/onnxruntime/js/…` overrides (see `.cursor/skills/ort-128-e2b-q4/SKILL.md`).
