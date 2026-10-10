/* global AudioWorkletProcessor, registerProcessor */

// Runs on the audio thread. Collects microphone samples and hands them to the page in
// blocks, instead of one tiny message per 128-sample render quantum.
const BLOCK = 2048;

class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.block = new Float32Array(BLOCK);
    this.filled = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) {
      for (let i = 0; i < channel.length; i += 1) {
        this.block[this.filled] = channel[i];
        this.filled += 1;
        if (this.filled === BLOCK) {
          this.port.postMessage(this.block.slice());
          this.filled = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("mic-capture", MicCapture);
