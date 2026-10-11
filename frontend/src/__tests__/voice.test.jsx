import {
  initialVoice,
  microphoneError,
  voiceReducer,
  voiceStatus,
} from "../components/voiceConversationMachine";
import { VoiceStream } from "../voice/VoiceStream";
import { downsample, floatToPcm16, pcm16ToFloat } from "../voice/pcm";

// --- PCM helpers ------------------------------------------------------------------------

describe("pcm", () => {
  test("16 kHz audio is passed through untouched", () => {
    const samples = new Float32Array([0.1, 0.2, 0.3]);

    expect(downsample(samples, 16000)).toBe(samples);
  });

  test.each([
    [48000, 4800, 1600],
    [44100, 4410, 1600],
  ])("%i Hz is brought down to 16 kHz", (rate, length, expected) => {
    expect(downsample(new Float32Array(length), rate)).toHaveLength(expected);
  });

  test("downsampling keeps the shape of the signal", () => {
    const ramp = Float32Array.from({ length: 480 }, (_, i) => i / 479);

    const out = downsample(ramp, 48000);

    expect(out[0]).toBeCloseTo(0);
    expect(out.at(-1)).toBeCloseTo(1);
    expect(out[80]).toBeCloseTo(0.5, 1);
  });

  test("floats become 16-bit samples, clipped rather than wrapped", () => {
    expect([...floatToPcm16(new Float32Array([0, 1, -1, 2, -2, 0.5]))]).toEqual([
      0, 32767, -32767, 32767, -32767, 16384,
    ]);
  });

  test("PCM16 bytes are read as little-endian", () => {
    const bytes = new Uint8Array([0x00, 0x40, 0x00, 0xc0]).buffer; // 16384, -16384

    expect([...pcm16ToFloat(bytes)]).toEqual([0.5, -0.5]);
  });
});

// --- the state machine ------------------------------------------------------------------

describe("voice state machine", () => {
  const on = voiceReducer(voiceReducer(initialVoice, { type: "starting" }), { type: "ready" });

  test("is off, with nothing to say, until started", () => {
    expect(voiceStatus(initialVoice)).toBe("");
  });

  test("starting, then ready", () => {
    expect(voiceStatus(voiceReducer(initialVoice, { type: "starting" }))).toMatch(/microphone/);
    expect(on.phase).toBe("on");
  });

  test.each([
    ["IDLE", /Speak whenever/],
    ["LISTENING", /Hearing you/],
    ["TRANSCRIBING", /what you said/],
    ["THINKING", /Thinking/],
    ["TOOL_EXECUTION", /Checking/],
    ["GENERATING_RESPONSE", /Thinking/],
    ["SPEAKING", /interrupt/],
    ["WAITING_FOR_NEXT_INPUT", /Speak whenever/],
  ])("server state %s is shown as %s", (state, expected) => {
    expect(voiceStatus(voiceReducer(on, { type: "state", state }))).toMatch(expected);
  });

  test("a server error is shown, and cleared once the patient is heard again", () => {
    const failed = voiceReducer(on, { type: "error", message: "Sorry, I could not hear that." });
    expect(failed.error).toBe("Sorry, I could not hear that.");

    const waiting = voiceReducer(failed, { type: "state", state: "WAITING_FOR_NEXT_INPUT" });
    expect(waiting.error).toBe("Sorry, I could not hear that.");
    expect(voiceReducer(waiting, { type: "state", state: "LISTENING" }).error).toBe("");
  });

  test.each([
    [4401, /log in again/],
    [4429, /wait a minute/],
    [1006, /closed/],
  ])("a connection closed with %i turns voice off and says why", (code, expected) => {
    const closed = voiceReducer(on, { type: "closed", code });

    expect(closed.phase).toBe("off");
    expect(closed.error).toMatch(expected);
  });

  test("stopping clears everything", () => {
    expect(voiceReducer(on, { type: "stopped" })).toEqual(initialVoice);
  });

  test("microphone failures are explained in plain words", () => {
    expect(microphoneError({ name: "NotAllowedError" })).toMatch(/blocked/);
    expect(microphoneError({ name: "NotFoundError" })).toMatch(/No microphone/);
    expect(microphoneError(new Error("boom"))).toMatch(/Could not start/);
  });
});

// --- VoiceStream ------------------------------------------------------------------------

class FakeSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.sent = [];
    FakeSocket.last = this;
  }
  open() {
    this.readyState = 1;
    this.onopen();
  }
  send(data) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
  }
  json() {
    return this.sent.filter((d) => typeof d === "string").map((d) => JSON.parse(d));
  }
  server(event) {
    this.onmessage({ data: typeof event === "string" ? event : JSON.stringify(event) });
  }
}

function harness({ micError } = {}) {
  const calls = { player: [], events: [], micStopped: 0 };
  let onChunk, onDrained;
  const devices = {
    WebSocket: FakeSocket,
    url: (pipeline) => `ws://test/ws/voice?pipeline=${pipeline}`,
    createMic: async (callback) => {
      if (micError) throw micError;
      onChunk = callback;
      return { stop: () => (calls.micStopped += 1) };
    },
    createPlayer: (drained) => {
      onDrained = drained;
      return {
        start: (rate) => calls.player.push(["start", rate]),
        enqueue: (buffer) => calls.player.push(["enqueue", buffer.byteLength]),
        finish: () => calls.player.push(["finish"]),
        stop: () => calls.player.push(["stop"]),
        close: () => calls.player.push(["close"]),
      };
    },
  };
  const stream = new VoiceStream({
    token: "jwt-token",
    sessionId: "text-session-1",
    onEvent: (event) => calls.events.push(event),
    devices,
  });
  return { stream, calls, speak: (chunk) => onChunk(chunk), drained: () => onDrained() };
}

describe("VoiceStream", () => {
  test("authenticates first, with the token in a message and never in the URL", async () => {
    const { stream } = harness();
    await stream.start();
    const socket = FakeSocket.last;
    socket.open();

    expect(socket.url).not.toContain("jwt-token");
    expect(socket.url).toBe("ws://test/ws/voice?pipeline=live");
    expect(socket.json()).toEqual([
      { type: "auth", token: "jwt-token", session_id: "text-session-1" },
    ]);
  });

  test("sends no microphone audio until the server says ready", async () => {
    const { stream, speak } = harness();
    await stream.start();
    const socket = FakeSocket.last;
    socket.open();

    speak(new Int16Array(4));
    expect(socket.sent.filter((d) => typeof d !== "string")).toHaveLength(0);

    socket.server({ type: "ready", session_id: "s", sample_rate: 16000 });
    speak(new Int16Array(4));
    expect(socket.sent.filter((d) => typeof d !== "string")).toHaveLength(1);
  });

  test("plays the reply and tells the server when playback has finished", async () => {
    const { stream, calls, drained } = harness();
    await stream.start();
    const socket = FakeSocket.last;
    socket.open();

    socket.server({ type: "audio_start", sample_rate: 22050 });
    socket.onmessage({ data: new ArrayBuffer(8) });
    socket.server({ type: "audio_end" });
    expect(calls.player).toEqual([["start", 22050], ["enqueue", 8], ["finish"]]);
    expect(socket.json().map((m) => m.type)).not.toContain("playback_done");

    drained();
    expect(socket.json().at(-1)).toEqual({ type: "playback_done" });
  });

  test("barge-in from the server stops playback at once", async () => {
    const { stream, calls } = harness();
    await stream.start();
    FakeSocket.last.open();

    FakeSocket.last.server({ type: "stop_audio" });

    expect(calls.player).toEqual([["stop"]]);
  });

  test("every server event is passed on, and junk is ignored", async () => {
    const { stream, calls } = harness();
    await stream.start();
    FakeSocket.last.open();

    FakeSocket.last.server("not json");
    FakeSocket.last.server({ type: "transcript", text: "hello", language: "en" });

    expect(calls.events).toEqual([{ type: "transcript", text: "hello", language: "en" }]);
  });

  test("stopping releases the microphone, the speakers and the connection", async () => {
    const { stream, calls } = harness();
    await stream.start();
    const socket = FakeSocket.last;
    socket.open();

    stream.stop();
    stream.stop();

    expect(calls.micStopped).toBe(1);
    expect(calls.player).toEqual([["close"]]);
    expect(socket.json().at(-1)).toEqual({ type: "end" });
    expect(socket.readyState).toBe(3);
  });

  test("a refused microphone opens no connection and leaves nothing running", async () => {
    FakeSocket.last = null;
    const denied = Object.assign(new Error("denied"), { name: "NotAllowedError" });
    const { stream, calls } = harness({ micError: denied });

    await expect(stream.start()).rejects.toBe(denied);

    expect(FakeSocket.last).toBeNull();
    expect(calls.player).toEqual([["close"]]);
  });

  test("a connection the server closes is reported once and cleaned up", async () => {
    const { stream, calls } = harness();
    await stream.start();
    const socket = FakeSocket.last;
    socket.open();

    socket.onclose({ code: 4429 });

    expect(calls.events).toEqual([{ type: "closed", code: 4429 }]);
    expect(calls.micStopped).toBe(1);
  });
});
