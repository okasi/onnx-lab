import { dtypeSuffix } from './models.mjs';

export const GEMMA4_MODELS = [
  {
    id: 'onnx-community/gemma-4-E2B-it-ONNX',
    slug: 'E2B-it',
    name: 'Gemma 4 E2B IT',
    quants: ['fp32', 'fp16', 'q4', 'q4f16', 'q8'],
  },
  {
    id: 'onnx-community/gemma-4-E4B-it-ONNX',
    slug: 'E4B-it',
    name: 'Gemma 4 E4B IT',
    quants: ['fp32', 'fp16', 'q4', 'q4f16', 'q8'],
  },
  {
    id: 'onnx-community/gemma-4-E2B-it-qat-mobile-ONNX',
    slug: 'E2B-qat-mobile',
    name: 'Gemma 4 E2B IT QAT Mobile',
    quants: ['q2f16'],
  },
  {
    id: 'onnx-community/gemma-4-E4B-it-qat-mobile-ONNX',
    slug: 'E4B-qat-mobile',
    name: 'Gemma 4 E4B IT QAT Mobile',
    quants: ['q2f16'],
  },
];

export const GEMMA4_BACKENDS = ['cpu', 'wasm-jsep', 'wasm', 'webgpu'];

export const GEMMA4_DEFAULT_PROMPT = 'Hello, my name is';
export const GEMMA4_DEFAULT_MAX_NEW_TOKENS = 8;

export function gemma4Suffix(dtype) {
  return dtypeSuffix(dtype);
}

export function findGemma4Model(idOrSlug) {
  return GEMMA4_MODELS.find(
    (m) => m.id === idOrSlug || m.slug === idOrSlug || m.id.endsWith(idOrSlug),
  );
}
