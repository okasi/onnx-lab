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
    note: 'Text encoder only; vision/audio ONNX files are not benchmarked here.',
  },
  {
    id: 'onnx-community/granite-embedding-311m-multilingual-r2-ONNX',
    name: 'Granite Embedding 311M Multilingual R2',
    url: 'https://huggingface.co/onnx-community/granite-embedding-311m-multilingual-r2-ONNX/tree/main/onnx',
  },
  {
    id: 'onnx-community/embeddinggemma-300m-ONNX',
    name: 'EmbeddingGemma 300M',
    url: 'https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/tree/main/onnx',
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
  },
  {
    id: 'Snowflake/snowflake-arctic-embed-l-v2.0',
    name: 'Snowflake Arctic Embed L v2.0',
    url: 'https://huggingface.co/Snowflake/snowflake-arctic-embed-l-v2.0/tree/main/onnx',
    extra_variants: [
      {
        label: 'O4',
        dtype: 'fp32',
        model_file_name: 'model_O4',
        note: 'Snowflake O4 quantized ONNX (non-standard dtype suffix).',
      },
    ],
  },
  {
    id: 'Snowflake/snowflake-arctic-embed-m-v2.0',
    name: 'Snowflake Arctic Embed M v2.0',
    url: 'https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v2.0/tree/main/onnx',
  },
];

/** Standard dtypes exposed by ModelRegistry for most ONNX embedding repos. */
export const STANDARD_DTYPES = [
  'fp32',
  'fp16',
  'int8',
  'uint8',
  'q8',
  'q4',
  'q4f16',
  'bnb4',
];
