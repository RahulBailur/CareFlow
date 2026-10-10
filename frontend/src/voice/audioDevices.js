// The two pieces that touch real browser audio. Kept apart from VoiceStream so that the
// protocol logic can be tested without a microphone or speakers.

import { TARGET_RATE, downsample, floatToPcm16, pcm16ToFloat } from "./pcm";

/** Start the microphone. `onChunk` receives Int16Array blocks of 16 kHz mono audio. */
export async function createMic(onChunk) {
  const stream = await navigator.mediaDevices.getUserMedia({
    // Echo cancellation matters: without it CareBot hears itself and interrupts itself
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    },
  });
  let context;
  try {
    context = new AudioContext({ sampleRate: TARGET_RATE });
  } catch {
    context = new AudioContext(); // the browser's own rate; chunks are downsampled below
  }
  await context.audioWorklet.addModule("/mic-worklet.js");
  const source = context.createMediaStreamSource(stream);
  const capture = new AudioWorkletNode(context, "mic-capture");
  capture.port.onmessage = (event) =>
    onChunk(floatToPcm16(downsample(event.data, context.sampleRate)));
  // A node only runs while it leads to the output, so route it there silently
  const mute = context.createGain();
  mute.gain.value = 0;
  source.connect(capture).connect(mute).connect(context.destination);

  return {
    stop() {
      capture.port.onmessage = null;
      source.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      context.close();
    },
  };
}

/** Plays PCM16 chunks back to back. `onDrained` fires when the last one has finished. */
export function createPlayer(onDrained) {
  const context = new AudioContext();
  const playing = new Set();
  let sampleRate = 24000;
  let nextStart = 0;
  let finishing = false;

  function checkDrained() {
    if (finishing && playing.size === 0) {
      finishing = false;
      onDrained();
    }
  }

  function stop() {
    finishing = false;
    playing.forEach((source) => {
      source.onended = null;
      try {
        source.stop();
      } catch {
        /* it had not started yet */
      }
    });
    playing.clear();
    nextStart = 0;
  }

  return {
    start(rate) {
      sampleRate = rate;
      finishing = false;
      context.resume();
    },
    enqueue(arrayBuffer) {
      const samples = pcm16ToFloat(arrayBuffer);
      if (samples.length === 0) return;
      const buffer = context.createBuffer(1, samples.length, sampleRate);
      buffer.copyToChannel(samples, 0);
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.connect(context.destination);
      const startAt = Math.max(context.currentTime + 0.02, nextStart);
      source.start(startAt);
      nextStart = startAt + buffer.duration;
      playing.add(source);
      source.onended = () => {
        playing.delete(source);
        checkDrained();
      };
    },
    /** No more audio is coming for this reply. */
    finish() {
      finishing = true;
      checkDrained();
    },
    stop,
    close() {
      stop();
      context.close();
    },
  };
}
