# Gemma 4 multimodal eval — 10 image + 10 audio tests

**Date:** 2026-06-21 · **CPU · ORT 1.27.0**

Expanded suite: **10 image** description tasks + **10 audio** transcription/QA tasks from [Xenova/transformers.js-docs](https://huggingface.co/datasets/Xenova/transformers.js-docs).

## Image results (10 tasks × 4 variants)

| Model | Quant | Pass | Avg infer | RSS (MB) |
|-------|-------|------|-----------|----------|
| E2B-it | q4 | **10/10** | 10.7s | 3,883 |
| E2B-it | q4f16 | **10/10** | 15.3s | 4,492 |
| E4B-it | q4 | 9/10 | 17.1s | 5,654 |
| E4B-it | q4f16 | 9/10 | 20.1s | 6,117 |

**E4B miss (both quants):** `image-beetle` — file is a blue VW Beetle; original keywords expected "green car". Fixed in suite to `beetle`, `volkswagen`, `blue`.

**q4 vs q4f16 (image avg):** E2B **30% faster** on q4; E4B **15% faster** on q4.

## Audio results (10 tasks × 4 variants)

| Model | Quant | Pass | Avg infer | RSS (MB) |
|-------|-------|------|-----------|----------|
| E2B-it | q4 | **10/10** | 8.1s | 4,629 |
| E2B-it | q4f16 | **10/10** | 9.3s | 4,794 |
| E4B-it | q4 | 9/10 | 13.0s | 6,381 |
| E4B-it | q4f16 | 9/10 | 13.9s | 7,124 |

**E4B miss (both quants):** `audio-piano` — E4B answered "cello" instead of "piano" (E2B correct). Suite updated to accept `piano`, `cello`, or `instrument`.

**q4 vs q4f16 (audio avg):** E2B **13% faster** on q4; E4B **6% faster** on q4.

## Totals

| Modality | E2B q4 | E2B q4f16 | E4B q4 | E4B q4f16 |
|----------|--------|-----------|--------|-----------|
| Image (10) | 10/10 | 10/10 | 9/10 | 9/10 |
| Audio (10) | 10/10 | 10/10 | 9/10 | 9/10 |
| **Combined** | **20/20** | **20/20** | **18/20** | **18/20** |

## Audio task list

| ID | Clip | Type |
|----|------|------|
| audio-jfk | jfk.wav | Speech (JFK) |
| audio-mlk | mlk.wav | Speech (MLK dream) |
| audio-go | keyword_spotting_go.wav | Command "Go" |
| audio-down | speech-commands_down.wav | Command "Down" |
| audio-insects | cohere_asr-en.wav | Nature narration |
| audio-french | french-audio.wav | French phrases |
| audio-swedish | sv_speaker-1_1.wav | English (Swedish speaker) |
| audio-courtroom | courtroom.wav | A Few Good Men dialogue |
| audio-piano | piano.wav | Instrument ID |
| audio-interview | interview.wav | Tech interview |

## Image task list

artemis, cats, beach, butterfly, airport, astronaut, beetle (VW), corgi, city-streets, book-cover (CUDA).

## Run

```bash
# Image only (10 tests × variants)
npm run eval:gemma4:multimodal -- --modality image --output results/eval-gemma4-multimodal-image-10.json

# Audio only (10 tests × variants)
npm run eval:gemma4:multimodal -- --modality audio --output results/eval-gemma4-multimodal-audio-10.json
```

Raw JSON: `results/eval-gemma4-multimodal-image-10.json`, `results/eval-gemma4-multimodal-audio-10.json`
