/**
 * ONNX embedding models to benchmark with Transformers.js (WASM in Node.js).
 * Hub links point at each repo's `onnx/` folder.
 */
export const MODELS = [
  {
    id: 'sirasagi62/granite-embedding-278m-multilingual-ONNX',
    name: 'Granite Embedding 278M Multilingual',
    url: 'https://huggingface.co/sirasagi62/granite-embedding-278m-multilingual-ONNX/tree/main/onnx',
  },
  {
    id: 'onnx-community/gte-multilingual-base',
    name: 'GTE Multilingual Base',
    url: 'https://huggingface.co/onnx-community/gte-multilingual-base/tree/main/onnx',
  },
  {
    id: 'onnx-community/jina-embeddings-v5-omni-nano-ONNX',
    name: 'Jina Embeddings v5 Omni Nano (text)',
    url: 'https://huggingface.co/onnx-community/jina-embeddings-v5-omni-nano-ONNX/tree/main/onnx',
    model_file_name: 'text_model',
    backend: 'cpu',
    note: 'Text encoder only; requires .onnx_data shards (CPU backend).',
  },
  {
    id: 'onnx-community/embeddinggemma-300m-ONNX',
    name: 'EmbeddingGemma 300M',
    url: 'https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/tree/main/onnx',
    backend: 'auto',
    note: 'q4/q4f16 need wasm-jsep + externalData mount; WebGPU needs shader-f16.',
    extra_variants: [
      {
        label: 'no_gather_q4',
        dtype: 'fp32',
        model_file_name: 'model_no_gather_q4',
        backend: 'cpu',
        note: 'Q4 weights without GatherBlockQuantized op (WASM-incompatible in standard q4).',
        when_dtype: 'q4',
      },
    ],
  },
  {
    id: 'onnx-community/Qwen3-Embedding-0.6B-ONNX',
    name: 'Qwen3 Embedding 0.6B',
    url: 'https://huggingface.co/onnx-community/Qwen3-Embedding-0.6B-ONNX/tree/main/onnx',
  },
  {
    id: 'onnx-community/bge-m3-ONNX',
    name: 'BGE-M3',
    url: 'https://huggingface.co/onnx-community/bge-m3-ONNX/tree/main/onnx',
    heavy_quants: ['bnb4', 'q4'],
    note: 'bnb4/q4 may OOM on WASM above ~10 GB RSS; auto-falls back to CPU.',
  },
];

export const BENCHMARK_DTYPES = ['bnb4', 'int8', 'q4', 'q4f16', 'q8', 'uint8'];

export const DTYPE_ALIASES = {
  quantized: 'q8',
  q8: 'q8',
};

export function normalizeDtype(name) {
  return DTYPE_ALIASES[name] ?? name;
}

export function dtypeLabel(dtype) {
  return dtype === 'q8' ? 'quantized (q8)' : dtype;
}

export function variantBackend(model, variant) {
  if (variant.backend) {
    return variant.backend;
  }
  if (model.backend) {
    return model.backend;
  }
  if (model.heavy_quants?.includes(variant.dtype)) {
    return 'auto';
  }
  return 'auto';
}
