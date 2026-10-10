// What the server takes from the microphone: 16 kHz mono PCM16.
export const TARGET_RATE = 16000;

/** Float32 samples at `fromRate` to Float32 at 16 kHz, by linear interpolation. */
export function downsample(samples, fromRate) {
  if (fromRate === TARGET_RATE) return samples;
  const length = Math.round((samples.length * TARGET_RATE) / fromRate);
  const out = new Float32Array(length);
  const step = (samples.length - 1) / Math.max(1, length - 1);
  for (let i = 0; i < length; i += 1) {
    const position = i * step;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    out[i] = samples[left] + (samples[right] - samples[left]) * (position - left);
  }
  return out;
}

export function floatToPcm16(samples) {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    out[i] = Math.round(clamped * 32767);
  }
  return out;
}

/** An ArrayBuffer of little-endian PCM16 to Float32 samples, ready for an AudioBuffer. */
export function pcm16ToFloat(buffer) {
  const view = new DataView(buffer);
  const out = new Float32Array(Math.floor(buffer.byteLength / 2));
  for (let i = 0; i < out.length; i += 1) out[i] = view.getInt16(i * 2, true) / 32768;
  return out;
}
