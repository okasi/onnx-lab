/**
 * Gemma 4 multimodal inference helpers (image + audio, CPU).
 */
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  AutoProcessor,
  Gemma4ForConditionalGeneration,
  load_image,
  env,
} from '@huggingface/transformers';
import { loadWavAudio } from './gemma4-audio-node.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, '..');

function configureEnv() {
  env.allowRemoteModels = true;
  env.useFSCache = true;
  env.cacheDir = path.join(root, '.cache', 'transformers-node');
}

/**
 * @param {string} modelId
 * @param {string} dtype
 */
export async function loadGemma4Multimodal(modelId, dtype) {
  configureEnv();
  const processor = await AutoProcessor.from_pretrained(modelId);
  const model = await Gemma4ForConditionalGeneration.from_pretrained(modelId, {
    dtype,
    device: 'cpu',
    session_options: {
      enableCpuMemArena: false,
      enableMemPattern: false,
    },
  });
  return { processor, model };
}

/**
 * @param {object} params
 * @param {import('@huggingface/transformers').Gemma4Processor} params.processor
 * @param {import('@huggingface/transformers').Gemma4ForConditionalGeneration} params.model
 * @param {'image'|'audio'} params.modality
 * @param {string} params.promptText
 * @param {string} params.mediaUrl
 * @param {number} params.maxNewTokens
 */
export async function runGemma4MultimodalTask({
  processor,
  model,
  modality,
  promptText,
  mediaUrl,
  maxNewTokens,
}) {
  const content = modality === 'image'
    ? [{ type: 'image' }, { type: 'text', text: promptText }]
    : [{ type: 'audio' }, { type: 'text', text: promptText }];

  const messages = [{ role: 'user', content }];
  const prompt = processor.apply_chat_template(messages, {
    enable_thinking: false,
    add_generation_prompt: true,
  });

  let image = null;
  let audio = null;
  if (modality === 'image') {
    image = await load_image(mediaUrl);
  } else {
    audio = await loadWavAudio(mediaUrl);
  }

  const inputs = await processor(prompt, image, audio, { add_special_tokens: false });
  const outputs = await model.generate({
    ...inputs,
    max_new_tokens: maxNewTokens,
    do_sample: false,
  });

  const decoded = processor.batch_decode(
    outputs.slice(null, [inputs.input_ids.dims.at(-1), null]),
    { skip_special_tokens: true },
  );

  return {
    generated_text: decoded[0] ?? '',
    prompt_chars: prompt.length,
  };
}
