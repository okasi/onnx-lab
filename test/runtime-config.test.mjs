import assert from 'node:assert/strict';
import test from 'node:test';
import { dtypeSuffix, normalizeDtype, variantBackend } from '../config/models.mjs';
import { gemma4WasmExternalData } from '../lib/transformers-runtime.mjs';

test('dtype and backend configuration has one canonical mapping', () => {
  assert.equal(normalizeDtype('quantized'), 'q8');
  assert.equal(dtypeSuffix('q8'), '_quantized');
  assert.equal(dtypeSuffix('custom'), '_custom');
  assert.equal(variantBackend({}, {}), 'auto');
  assert.equal(variantBackend({ backend: 'cpu' }, {}), 'cpu');
  assert.equal(variantBackend({ backend: 'cpu' }, { backend: 'wasm' }), 'wasm');
});

test('Gemma external data follows configured shard counts', () => {
  assert.deepEqual(gemma4WasmExternalData({}, 'q4'), []);
  assert.deepEqual(
    gemma4WasmExternalData({
      use_external_data_format: {
        'embed_tokens_q4.onnx': 2,
        decoder_model_merged: 1,
      },
    }, 'q4'),
    [
      { path: 'embed_tokens_q4.onnx_data', data: 'onnx/embed_tokens_q4.onnx_data' },
      { path: 'embed_tokens_q4.onnx_data_1', data: 'onnx/embed_tokens_q4.onnx_data_1' },
      {
        path: 'decoder_model_merged_q4.onnx_data',
        data: 'onnx/decoder_model_merged_q4.onnx_data',
      },
    ],
  );
});
