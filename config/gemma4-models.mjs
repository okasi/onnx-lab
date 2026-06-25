/**
 * Gemma 4 ONNX text-generation models (Gemma4ForCausalLM — text-only sessions).
 * Hub repos expose flat `onnx/` files: embed_tokens* + decoder_model_merged* per quant.
 */
export const GEMMA4_MODELS = [
  {
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    slug: 'E2B-it',
    name: 'Gemma 4 E2B IT',
    url: 'https://huggingface.co/onnx-community/gemma-4-E2B-it-ONNX/tree/main/onnx',
    quants: ['fp32', 'fp16', 'q4', 'q4f16', 'q8'],
    note: '2B multimodal (text+image+audio); q8 uses _quantized ONNX suffix.',
  },
  {
    id: 'onnx-community/gemma-4-E4B-it-ONNX',
    slug: 'E4B-it',
    name: 'Gemma 4 E4B IT',
    url: 'https://huggingface.co/onnx-community/gemma-4-E4B-it-ONNX/tree/main/onnx',
    quants: ['fp32', 'fp16', 'q4', 'q4f16', 'q8'],
    note: '4B multimodal (text+image+audio); large downloads and high RAM.',
  },
  {
    id: 'onnx-community/gemma-4-E2B-it-qat-mobile-ONNX',
    slug: 'E2B-qat-mobile',
    name: 'Gemma 4 E2B IT QAT Mobile',
    url: 'https://huggingface.co/onnx-community/gemma-4-E2B-it-qat-mobile-ONNX/tree/main/onnx',
    quants: ['q2f16'],
    note: 'Mobile QAT — q2f16 only; needs ORT 1.27+ for GatherBlockQuantized bits=2 (see docs/gemma4-q2f16.md).',
  },
  {
    id: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
    slug: 'E4B-qat-mobile',
    name: 'Gemma 4 E4B IT QAT Mobile',
    url: 'https://huggingface.co/onnx-community/gemma-4-E4B-it-qat-mobile-ONNX/tree/main/onnx',
    quants: ['q2f16'],
    note: 'Mobile QAT — q2f16 only.',
  },
];

/** ONNX filename suffix per Transformers.js dtype (text sessions). */
export const GEMMA4_DTYPE_SUFFIX = {
  fp32: '',
  fp16: '_fp16',
  q8: '_quantized',
  q4: '_q4',
  q4f16: '_q4f16',
  q2f16: '_q2f16',
};

export const GEMMA4_BACKENDS = ['cpu', 'wasm-jsep', 'wasm', 'webgpu'];

export const GEMMA4_DEFAULT_PROMPT = 'Hello, my name is';
export const GEMMA4_DEFAULT_MAX_NEW_TOKENS = 8;

export function gemma4Suffix(dtype) {
  return GEMMA4_DTYPE_SUFFIX[dtype] ?? `_${dtype}`;
}

export function findGemma4Model(idOrSlug) {
  return GEMMA4_MODELS.find(
    (m) => m.id === idOrSlug || m.slug === idOrSlug || m.id.endsWith(idOrSlug),
  );
}
