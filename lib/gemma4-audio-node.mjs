/**
 * Load remote/local WAV audio for Gemma 4 in Node (no AudioContext).
 */
import wavefile from 'wavefile';

const { WaveFile } = wavefile;

const DEFAULT_SAMPLE_RATE = 16000;

function mergeChannels(samples) {
  if (!Array.isArray(samples)) {
    return samples;
  }
  if (samples.length === 1) {
    return samples[0];
  }
  const scaling = Math.sqrt(2);
  const merged = samples[0];
  for (let i = 0; i < merged.length; i += 1) {
    merged[i] = (scaling * (samples[0][i] + samples[1][i])) / 2;
  }
  return merged;
}

/**
 * @param {string} urlOrPath HTTP(S) URL or local file path
 * @param {number} [sampleRate=16000]
 * @returns {Promise<Float32Array>}
 */
export async function loadWavAudio(urlOrPath, sampleRate = DEFAULT_SAMPLE_RATE) {
  let buffer;
  if (urlOrPath.startsWith('http://') || urlOrPath.startsWith('https://')) {
    const res = await fetch(urlOrPath);
    if (!res.ok) {
      throw new Error(`Failed to fetch audio: ${res.status} ${urlOrPath}`);
    }
    buffer = Buffer.from(await res.arrayBuffer());
  } else {
    const fs = await import('node:fs/promises');
    buffer = await fs.readFile(urlOrPath);
  }

  const wav = new WaveFile(buffer);
  wav.toBitDepth('32f');
  wav.toSampleRate(sampleRate);
  const samples = wav.getSamples();
  const mono = mergeChannels(samples);
  if (!(mono instanceof Float32Array)) {
    return new Float32Array(mono);
  }
  return mono;
}
