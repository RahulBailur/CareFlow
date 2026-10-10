import { createMic, createPlayer } from "./audioDevices";

function voiceUrl(pipeline) {
  const scheme = window.location.protocol === "https:" ? "wss" : "ws";
  return `${scheme}://${window.location.host}/ws/voice?pipeline=${pipeline}`;
}

/**
 * One voice conversation: microphone up, replies down, over a single WebSocket.
 *
 * Every server event is passed to `onEvent`; this class additionally acts on the ones about
 * audio (start, stop for barge-in, end) and tells the server when playback has finished.
 */
export class VoiceStream {
  constructor({ token, sessionId, onEvent, pipeline = "live", devices = {} }) {
    this.token = token;
    this.pipeline = pipeline;
    this.sessionId = sessionId;
    this.onEvent = onEvent;
    this.devices = {
      createMic,
      createPlayer,
      WebSocket: window.WebSocket,
      url: voiceUrl,
      ...devices,
    };
    this.ready = false;
    this.stopped = false;
  }

  /** Must be called from a click, so the browser allows the microphone and audio output. */
  async start() {
    this.player = this.devices.createPlayer(() => this.send({ type: "playback_done" }));
    try {
      this.mic = await this.devices.createMic((chunk) => {
        // Nothing is sent until the server has accepted the token
        if (this.ready && this.socket?.readyState === 1) this.socket.send(chunk.buffer);
      });
    } catch (error) {
      this.stop();
      throw error;
    }
    if (this.stopped) return; // stopped while the permission prompt was open

    const socket = new this.devices.WebSocket(this.devices.url(this.pipeline));
    socket.binaryType = "arraybuffer";
    socket.onopen = () =>
      this.send({
        type: "auth",
        token: this.token,
        ...(this.sessionId && { session_id: this.sessionId }),
      });
    socket.onmessage = (message) => this.receive(message.data);
    socket.onclose = (event) => {
      if (this.stopped) return;
      this.onEvent({ type: "closed", code: event.code });
      this.stop();
    };
    this.socket = socket;
  }

  send(message) {
    if (this.socket?.readyState === 1) this.socket.send(JSON.stringify(message));
  }

  receive(data) {
    if (typeof data !== "string") {
      this.player.enqueue(data);
      return;
    }
    let event;
    try {
      event = JSON.parse(data);
    } catch {
      return;
    }
    if (event.type === "ready") this.ready = true;
    else if (event.type === "audio_start") this.player.start(event.sample_rate);
    else if (event.type === "audio_end") this.player.finish();
    else if (event.type === "stop_audio") this.player.stop();
    this.onEvent(event);
  }

  stop() {
    if (this.stopped) return;
    this.stopped = true;
    this.ready = false;
    this.send({ type: "end" });
    this.mic?.stop();
    this.player?.close();
    this.socket?.close();
  }
}
