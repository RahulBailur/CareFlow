// What the voice UI shows. Driven only by events from the server and the device, never by
// guessed timers: if the status says "Thinking", the server is thinking.

export const initialVoice = { phase: "off", serverState: null, error: "" };

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
      return { phase: "starting", serverState: null, error: "" };
    case "ready":
      return { ...state, phase: "on" };
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
