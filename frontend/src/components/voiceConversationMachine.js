// What the voice UI shows. Driven only by events from the server and the device, never by
// guessed timers: if the status says "Thinking", the server is thinking.

// `pipeline` is the one the server says is actually running, which is not always the one
// asked for: Pipeline A falls back to B when no real-time session can be had.
export const initialVoice = {
  phase: "off",
  serverState: null,
  error: "",
  pipeline: null,
  notice: "",
};

export const PIPELINES = {
  live: "Real-time (A)",
  cascade: "Cascaded (B)",
};

const FELL_BACK = "Real-time voice is not available right now, so this is the cascaded pipeline.";
const SWITCHED = "The real-time connection was lost. Carrying on with the cascaded pipeline.";

const STATUS = {
  IDLE: "Listening. Speak whenever you are ready.",
  WAITING_FOR_NEXT_INPUT: "Listening. Speak whenever you are ready.",
  LISTENING: "Hearing you…",
  TRANSCRIBING: "Working out what you said…",
  THINKING: "Thinking…",
  TOOL_EXECUTION: "Checking…",
  GENERATING_RESPONSE: "Thinking…",
  SPEAKING: "Speaking. Talk over me to interrupt.",
  ERROR: "Something went wrong.",
};

const CLOSE_REASONS = {
  4401: "Your session has expired. Please log in again.",
  4429: "Too many voice sessions. Please wait a minute and try again.",
};

export function microphoneError(error) {
  if (error?.name === "NotAllowedError") {
    return "Microphone access was blocked. Allow it in your browser's site settings and try again.";
  }
  if (error?.name === "NotFoundError") return "No microphone was found on this device.";
  return "Could not start the microphone.";
}

export function voiceReducer(state, event) {
  switch (event.type) {
    case "starting":
      return { ...initialVoice, phase: "starting" };
    case "ready":
      return {
        ...state,
        phase: "on",
        pipeline: event.pipeline ?? null,
        notice: event.fallback_reason ? FELL_BACK : "",
      };
    case "pipeline":
      return { ...state, pipeline: event.pipeline, notice: SWITCHED };
    case "state":
      if (event.state === "END") return state;
      // Hearing the patient again means the last problem is behind us
      return {
        ...state,
        serverState: event.state,
        error: event.state === "LISTENING" ? "" : state.error,
      };
    case "error":
      return { ...state, error: event.message };
    case "failed":
      return { ...initialVoice, error: event.message };
    case "closed":
      return {
        ...initialVoice,
        error:
          CLOSE_REASONS[event.code] ?? "The voice connection closed. Start voice to try again.",
      };
    case "stopped":
      return initialVoice;
    default:
      return state;
  }
}

/** The one line shown under the conversation while voice is on. */
export function voiceStatus(state) {
  if (state.phase === "starting") return "Starting the microphone…";
  if (state.phase !== "on") return "";
  return STATUS[state.serverState] ?? "Connecting…";
}
