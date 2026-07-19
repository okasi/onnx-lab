export const MODELS = [
  {
    id: 'sirasagi62/granite-embedding-278m-multilingual-ONNX',
    name: 'Granite Embedding 278M Multilingual',
  },
  {
    id: 'onnx-community/gte-multilingual-base',
    name: 'GTE Multilingual Base',
  },
  {
    id: 'onnx-community/jina-embeddings-v5-omni-nano-ONNX',
    name: 'Jina Embeddings v5 Omni Nano (text)',
    model_file_name: 'text_model',
    backend: 'cpu',
  },
  {
    id: 'onnx-community/embeddinggemma-300m-ONNX',
    name: 'EmbeddingGemma 300M',
    extra_variants: [
      {
        label: 'no_gather_q4',
        dtype: 'fp32',
        model_file_name: 'model_no_gather_q4',
        backend: 'cpu',
        when_dtype: 'q4',
      },
    ],
  },
  {
    id: 'onnx-community/Qwen3-Embedding-0.6B-ONNX',
    name: 'Qwen3 Embedding 0.6B',
  },
  {
    id: 'onnx-community/bge-m3-ONNX',
    name: 'BGE-M3',
  },
];

export const BENCHMARK_DTYPES = ['bnb4', 'int8', 'q4', 'q4f16', 'q8', 'uint8'];

export const DTYPE_SUFFIXES = {
  fp32: '',
  fp16: '_fp16',
  bnb4: '_bnb4',
  int8: '_int8',
  q2f16: '_q2f16',
  q4: '_q4',
  q4f16: '_q4f16',
  q8: '_quantized',
  uint8: '_uint8',
};

export function normalizeDtype(name) {
  return name === 'quantized' ? 'q8' : name;
}

export function dtypeLabel(dtype) {
  return dtype === 'q8' ? 'quantized (q8)' : dtype;
}

export function dtypeSuffix(dtype) {
  return DTYPE_SUFFIXES[dtype] ?? `_${dtype}`;
}

export function variantBackend(model, variant) {
  return variant.backend ?? model.backend ?? 'auto';
}
