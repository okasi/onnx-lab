# Gemma 4 multimodal eval — image & audio (CPU, ORT 1.27)

**Date:** 2026-06-21 · **Backend:** CPU · **ORT:** 1.27.0 npm

Evaluated **image description** and **audio transcription** on E2B-it and E4B-it with **q4** and **q4f16** using `Gemma4ForConditionalGeneration` (full multimodal stack: `vision_encoder` + `audio_encoder` + `embed_tokens` + `decoder_model_merged`).

## Tasks

| ID | Modality | Input | Prompt |
|----|----------|-------|--------|
| image-artemis | Image | [artemis.jpeg](https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/artemis.jpeg) | Describe this image in one sentence. |
| audio-jfk | Audio | [jfk.wav](https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/jfk.wav) | Transcribe the audio verbatim. |

Scoring: substring match on expected keywords (flag/rocket/launch for image; Americans/country for audio).

## Results summary

| Model | Quant | Status | Load | Image infer | Audio infer | Pass | RSS (MB) |
|-------|-------|--------|------|-------------|-------------|------|----------|
| E2B-it | q4 | ok | 3.9s | **10.9s** | **9.2s** | 2/2 | 2,946 |
| E2B-it | q4f16 | ok | 7.3s | 17.0s | 10.2s | 2/2 | 3,515 |
| E4B-it | q4 | ok | 10.8s | 17.4s | 14.4s | 2/2 | 5,236 |
| E4B-it | q4f16 | ok | 10.7s | 20.8s | 17.0s | 2/2 | 5,845 |

**8/8 tasks passed** across all four variants.

## q4 vs q4f16

| Comparison | Image | Audio |
|------------|-------|-------|
| E2B q4 vs q4f16 | **36% faster** (10.9s vs 17.0s) | **10% faster** (9.2s vs 10.2s) |
| E4B q4 vs q4f16 | **17% faster** (17.4s vs 20.8s) | **15% faster** (14.4s vs 17.0s) |

q4 is faster for multimodal on CPU; q4f16 uses less decoder memory in some setups but here RSS is slightly higher for q4f16.

## E2B vs E4B

| Modality | E2B q4 | E4B q4 | E4B / E2B |
|----------|--------|--------|-----------|
| Image | 10.9s | 17.4s | 1.6× slower |
| Audio | 9.2s | 14.4s | 1.6× slower |
| Peak RSS | 2.9 GB | 5.2 GB | 1.8× RAM |

E4B produces slightly richer captions (adds punctuation on JFK audio) but costs ~1.6× latency and ~1.8× RAM.

## Sample outputs

**E2B q4 image:** “A large American flag is billowing in the sky next to a rocket launch…”

**E4B q4 image:** “A bright rocket launches against a clear blue sky, with the American flag prominently displayed…”

**All variants audio (JFK):** Correct transcription of “ask not what your country can do for you…”

## Run again

```bash
npm install
npm run eval:gemma4:multimodal
# or:
node --expose-gc scripts/eval-gemma4-multimodal.mjs --model E2B-it,E4B-it --dtype q4,q4f16
```

Raw JSON: `results/eval-gemma4-multimodal-full.json`

## Notes

- Audio in Node uses `wavefile` to load WAV (no `AudioContext`) — see `lib/gemma4-audio-node.mjs`.
- Multimodal loads **all four ONNX sessions** per quant; first run downloads vision/audio shards (~100–190 MB each).
- Text-only benchmarks (`benchmark-gemma4`) do not exercise vision/audio encoders.
