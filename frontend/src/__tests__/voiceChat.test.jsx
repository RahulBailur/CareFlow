import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import ChatBot, { APPOINTMENTS_CHANGED } from "../components/ChatBot";

const auth = vi.hoisted(() => ({ user: { id: "p1", name: "Ananya", role: "patient" } }));
const fake = vi.hoisted(() => ({ streams: [], startError: null }));

vi.mock("../context/AuthContext", () => ({ useAuth: () => ({ user: auth.user }) }));
vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal()),
  default: { post: vi.fn() },
}));

vi.mock("../voice/VoiceStream", () => ({
  VoiceStream: class {
    constructor(options) {
      this.options = options;
      this.stopped = false;
      fake.streams.push(this);
    }
    async start() {
      if (fake.startError) throw fake.startError;
    }
    stop() {
      this.stopped = true;
    }
    emit(event) {
      this.options.onEvent(event);
    }
  },
}));

describe("voice in the chat window", () => {
  beforeEach(() => {
    fake.streams.length = 0;
    fake.startError = null;
    localStorage.setItem("careflow_token", "jwt-token");
  });

  async function startVoice() {
    const user = userEvent.setup();
    render(<ChatBot />);
    await user.click(screen.getByRole("button", { name: "Ask CareBot" }));
    await user.click(screen.getByRole("button", { name: "Start voice" }));
    const stream = fake.streams.at(-1);
    act(() => stream.emit({ type: "ready", session_id: "voice-session-9", sample_rate: 16000 }));
    return { user, stream };
  }

  test("voice is off until the patient starts it", async () => {
    const user = userEvent.setup();
    render(<ChatBot />);
    await user.click(screen.getByRole("button", { name: "Ask CareBot" }));

    expect(fake.streams).toHaveLength(0);
    expect(screen.getByRole("button", { name: "Start voice" })).toHaveAttribute(
      "aria-pressed",
      "false",
    );
    expect(screen.getByRole("status")).toBeEmptyDOMElement();
  });

  test("starting voice uses the stored token and shows the live status", async () => {
    const { stream } = await startVoice();

    expect(stream.options.token).toBe("jwt-token");
    expect(screen.getByRole("button", { name: "Stop voice" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );

    act(() => stream.emit({ type: "state", state: "THINKING" }));
    expect(screen.getByRole("status")).toHaveTextContent("Thinking…");
  });

  test("what was heard and what was answered appear in the conversation", async () => {
    const { stream } = await startVoice();

    act(() => {
      stream.emit({ type: "transcript", text: "मुझे बुखार है", language: "hi" });
      stream.emit({
        type: "reply",
        text: "जनरल मेडिसिन विभाग में जाइए।",
        language: "hi",
        disclaimer: "यह कोई निदान नहीं है।",
        tools_called: [],
      });
    });

    const log = screen.getByRole("log");
    expect(within(log).getByText("मुझे बुखार है")).toHaveAttribute("lang", "hi");
    expect(within(log).getByText("जनरल मेडिसिन विभाग में जाइए।")).toBeInTheDocument();
    expect(screen.getByRole("note")).toHaveTextContent("यह कोई निदान नहीं है।");
  });

  test("a booking made by voice tells the rest of the app to reload", async () => {
    const listener = vi.fn();
    window.addEventListener(APPOINTMENTS_CHANGED, listener);
    const { stream } = await startVoice();

    act(() =>
      stream.emit({ type: "reply", text: "Booked.", language: "en", tools_called: ["book_slot"] }),
    );

    expect(listener).toHaveBeenCalledTimes(1);
    window.removeEventListener(APPOINTMENTS_CHANGED, listener);
  });

  test("Stop voice releases the microphone", async () => {
    const { user, stream } = await startVoice();

    await user.click(screen.getByRole("button", { name: "Stop voice" }));

    expect(stream.stopped).toBe(true);
    expect(screen.getByRole("button", { name: "Start voice" })).toBeInTheDocument();
    expect(screen.getByRole("status")).toBeEmptyDOMElement();
  });

  test("closing the chat window also releases the microphone", async () => {
    const { user, stream } = await startVoice();

    await user.click(screen.getByRole("button", { name: "Close" }));

    expect(stream.stopped).toBe(true);
  });

  test("a blocked microphone is explained and voice stays off", async () => {
    fake.startError = Object.assign(new Error("denied"), { name: "NotAllowedError" });
    const user = userEvent.setup();
    render(<ChatBot />);
    await user.click(screen.getByRole("button", { name: "Ask CareBot" }));

    await user.click(screen.getByRole("button", { name: "Start voice" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Microphone access was blocked");
    expect(screen.getByRole("button", { name: "Start voice" })).toBeInTheDocument();
  });

  test("a connection the server closes turns voice off and says why", async () => {
    const { stream } = await startVoice();

    act(() => stream.emit({ type: "closed", code: 4429 }));

    expect(screen.getByRole("alert")).toHaveTextContent("Too many voice sessions");
    expect(screen.getByRole("button", { name: "Start voice" })).toBeInTheDocument();
  });

  test("the conversation started by voice is continued by text", async () => {
    const api = (await import("../api")).default;
    api.post.mockResolvedValue({
      data: { reply: "ok", session_id: "voice-session-9", language: "en", tools_called: [] },
    });
    const { user } = await startVoice();

    await user.type(screen.getByLabelText("Message to CareBot"), "and by text{Enter}");

    expect(api.post).toHaveBeenCalledWith("/api/chat", {
      message: "and by text",
      session_id: "voice-session-9",
    });
  });
});

describe("the voice pipeline toggle", () => {
  beforeEach(() => {
    fake.streams.length = 0;
    fake.startError = null;
  });

  async function open() {
    const user = userEvent.setup();
    render(<ChatBot />);
    await user.click(screen.getByRole("button", { name: "Ask CareBot" }));
    return user;
  }

  test("asks for the real-time pipeline unless the patient picks the other", async () => {
    const user = await open();
    await user.click(screen.getByRole("button", { name: "Start voice" }));
    expect(fake.streams.at(-1).options.pipeline).toBe("live");
    act(() => fake.streams.at(-1).emit({ type: "ready", session_id: "s-12345678" }));

    await user.click(screen.getByRole("button", { name: "Stop voice" }));
    await user.selectOptions(screen.getByLabelText("Voice pipeline"), "cascade");
    await user.click(screen.getByRole("button", { name: "Start voice" }));
    expect(fake.streams.at(-1).options.pipeline).toBe("cascade");
  });

  test("cannot be changed while voice is on", async () => {
    const user = await open();
    expect(screen.getByLabelText("Voice pipeline")).toBeEnabled();

    await user.click(screen.getByRole("button", { name: "Start voice" }));

    expect(screen.getByLabelText("Voice pipeline")).toBeDisabled();
  });

  test("shows the pipeline the server is really running, and says when it fell back", async () => {
    const user = await open();
    await user.click(screen.getByRole("button", { name: "Start voice" }));

    act(() =>
      fake.streams.at(-1).emit({
        type: "ready",
        session_id: "voice-session-9",
        pipeline: "cascade",
        fallback_reason: "quota",
      }),
    );

    expect(screen.getByLabelText("Voice pipeline")).toHaveValue("cascade");
    expect(screen.getByRole("status")).toHaveTextContent("Real-time voice is not available");
  });

  test("says so when the real-time connection is lost mid-conversation", async () => {
    const user = await open();
    await user.click(screen.getByRole("button", { name: "Start voice" }));
    const stream = fake.streams.at(-1);
    act(() => stream.emit({ type: "ready", session_id: "voice-session-9", pipeline: "live" }));
    expect(screen.getByLabelText("Voice pipeline")).toHaveValue("live");

    act(() => stream.emit({ type: "pipeline", pipeline: "cascade", reason: "dropped" }));

    expect(screen.getByLabelText("Voice pipeline")).toHaveValue("cascade");
    expect(screen.getByRole("status")).toHaveTextContent("real-time connection was lost");
  });
});
