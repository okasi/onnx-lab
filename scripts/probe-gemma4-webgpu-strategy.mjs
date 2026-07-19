#!/usr/bin/env node
import {
  GEMMA4_DEFAULT_MAX_NEW_TOKENS,
  GEMMA4_DEFAULT_PROMPT,
} from '../config/gemma4-models.mjs';
import {
  classifyRuntimeError,
} from '../lib/benchmark-support.mjs';
import {
  GEMMA4_WEBGPU_STRATEGIES,
  runGemma4BrowserStrategy,
} from '../lib/gemma4-webgpu.mjs';

const modelId = process.argv[2];
const strategy = process.argv[3];

async function main() {
  const spec = GEMMA4_WEBGPU_STRATEGIES[strategy];
  const result = {
    model_id: modelId,
    strategy,
    status: 'pending',
    load_ms: null,
    infer_ms: null,
    generated_text: null,
    shader_f16: null,
    dtype_used: spec?.dtype ?? null,
    error: null,
    error_kind: null,
  };
  if (!modelId || !spec) {
    result.status = 'error';
    result.error = !modelId ? 'model id required' : `unknown strategy: ${strategy}`;
    console.log(JSON.stringify(result));
    return;
  }

  try {
    const payload = await runGemma4BrowserStrategy({
      modelId,
      strategyName: strategy,
      prompts: [{ id: 'smoke', prompt: GEMMA4_DEFAULT_PROMPT }],
      maxNewTokens: GEMMA4_DEFAULT_MAX_NEW_TOKENS,
      timeoutMs: 3_600_000,
    });
    result.load_ms = payload.load_ms;
    result.infer_ms = payload.inference.mean_ms;
    result.generated_text = payload.outputs[0]?.generated_text ?? null;
    result.shader_f16 = payload.shader_f16;
    result.dtype_used = payload.dtype_used;
    result.total_ms = payload.total_ms;
    result.status = 'ok';
  } catch (error) {
    const message = (error instanceof Error ? error.message : String(error)).slice(0, 500);
    result.error = message;
    result.error_kind = classifyRuntimeError(message);
    result.status = result.load_ms == null ? 'error' : 'infer_error';
  }
  console.log(JSON.stringify(result));
}

main().catch((error) => {
  console.log(JSON.stringify({ status: 'error', error: error.message }));
  process.exit(1);
});
