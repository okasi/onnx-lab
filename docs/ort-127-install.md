# ONNX Runtime 1.27 setup

This repository pins `onnxruntime-common`, `onnxruntime-node`, and
`onnxruntime-web` to 1.27.0.

## Install

```bash
npm install
npm ls onnxruntime-node onnxruntime-web onnxruntime-common --all
```

The dependency tree should show Transformers.js deduplicated onto the top-level
1.27.0 packages.

The package also overrides ORT's older `adm-zip` range to 0.6.0, which preserves
the install API while resolving the current archive-allocation advisory.

`sharp` may choose a source build when it detects a global `libvips`
installation. The repository declares `node-addon-api` and `node-gyp` as
development dependencies so this clean-install path works as well as the normal
prebuilt-binary path.

## Verify versions

```bash
node -e "import('onnxruntime-node').then(m => console.log(m.env.versions))"
node -e "import('onnxruntime-web').then(m => console.log(m.env.versions))"
```

Expected common/node/web version: `1.27.0`.

## Runtime usage

```javascript
import {
  createFeatureExtractor,
  createTextGenerator,
} from './lib/transformers-runtime.mjs';

const embeddings = await createFeatureExtractor(
  'onnx-community/embeddinggemma-300m-ONNX',
  { dtype: 'q4' },
  'cpu',
);

const generator = await createTextGenerator(
  'onnx-community/gemma-4-E2B-it-ONNX',
  { dtype: 'q4' },
  'cpu',
);
```

Use `wasm-jsep`, not `wasm`, for graphs containing
`GatherBlockQuantized`.

## q2f16 verification

These commands require the mobile model file to already exist in
`.cache/transformers-node/`:

```bash
npm run verify:ort:q2f16
npm run verify:ort:web:q2f16
```

They fail with an explicit missing-file message when the model is not cached.

## Optional source build

The npm packages are the default. Build ORT main only when testing an unreleased
runtime change:

```bash
npm run build:ort
npm run build:ort:web
```

The scripts use `vendor/onnxruntime/`, detect CPU count and host platform, and
leave the generated source/build tree gitignored.
