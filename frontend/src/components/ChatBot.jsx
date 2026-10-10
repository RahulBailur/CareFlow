import { useCallback, useEffect, useReducer, useRef, useState } from "react";

import api, { errorMessage, tokenStore } from "../api";
import { useAuth } from "../context/AuthContext";
import { VoiceStream } from "../voice/VoiceStream";
import TriageDisclaimer from "./TriageDisclaimer";
import {
  initialVoice,
  microphoneError,
  voiceReducer,
  voiceStatus,
} from "./voiceConversationMachine";

const MAX_LENGTH = 500;
const CHANGING_TOOLS = ["book_slot", "cancel_appointment", "reschedule_appointment"];
export const APPOINTMENTS_CHANGED = "careflow:appointments-changed";

const WELCOME = {
  role: "bot",
  text: "Hi, I'm CareBot. I can book, change or cancel an appointment, suggest which department to visit, show your visit history, and answer questions about the hospital. You can write or speak in English, Hindi or Kannada.",
};

function announceChanges(toolsCalled) {
  if (toolsCalled?.some((tool) => CHANGING_TOOLS.includes(tool))) {
    window.dispatchEvent(new Event(APPOINTMENTS_CHANGED));
  }
}

/** CareBot for patients: a floating button that opens a conversation, by text or by voice. */
export default function ChatBot() {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([WELCOME]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [voice, dispatchVoice] = useReducer(voiceReducer, initialVoice);
  const sessionId = useRef(null);
  const stream = useRef(null);
  const inputRef = useRef(null);
  const launcherRef = useRef(null);
  const endRef = useRef(null);

  const addMessage = useCallback((message) => setMessages((current) => [...current, message]), []);

  const stopVoice = useCallback(() => {
    stream.current?.stop();
    stream.current = null;
    dispatchVoice({ type: "stopped" });
  }, []);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages, sending, open, voice.serverState]);

  // The microphone never outlives the panel or the login
  useEffect(() => () => stream.current?.stop(), []);

  if (user?.role !== "patient") return null;

  function close() {
    stopVoice();
    setOpen(false);
    // After the panel unmounts, so focus has somewhere to land
    setTimeout(() => launcherRef.current?.focus(), 0);
  }

  function onVoiceEvent(event) {
    if (event.type === "ready") {
      sessionId.current = event.session_id; // text and voice share one conversation
    } else if (event.type === "transcript") {
      addMessage({ role: "user", text: event.text, lang: event.language });
    } else if (event.type === "reply") {
      addMessage({
        role: "bot",
        text: event.text,
        lang: event.language,
        disclaimer: event.disclaimer,
      });
      announceChanges(event.tools_called);
    } else if (event.type === "closed") {
      stream.current = null;
    }
    dispatchVoice(event);
  }

  async function startVoice() {
    setError("");
    dispatchVoice({ type: "starting" });
    const voiceStream = new VoiceStream({
      token: tokenStore.get(),
      sessionId: sessionId.current,
      onEvent: onVoiceEvent,
    });
    stream.current = voiceStream;
    try {
      await voiceStream.start();
    } catch (err) {
      stream.current = null;
      dispatchVoice({ type: "failed", message: microphoneError(err) });
    }
  }

  async function send(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sending) return;
    addMessage({ role: "user", text });
    setDraft("");
    setError("");
    setSending(true);
    try {
      const { data } = await api.post("/api/chat", {
        message: text,
        ...(sessionId.current && { session_id: sessionId.current }),
      });
      sessionId.current = data.session_id;
      addMessage({
        role: "bot",
        text: data.reply,
        lang: data.language,
        disclaimer: data.disclaimer,
      });
      announceChanges(data.tools_called);
    } catch (err) {
      setError(errorMessage(err, "CareBot could not answer. Please try again."));
      setDraft(text); // nothing is lost: the message goes back into the box
    } finally {
      setSending(false);
      inputRef.current?.focus();
    }
  }

  if (!open) {
    return (
      <button
        type="button"
        className="chat-launcher button primary"
        ref={launcherRef}
        onClick={() => setOpen(true)}
      >
        Ask CareBot
      </button>
    );
  }

  const voiceOn = voice.phase !== "off";
  const status = voiceStatus(voice);
  const problem = error || voice.error;

  return (
    <section
      className="chat-panel"
      aria-label="CareBot chat"
      onKeyDown={(event) => event.key === "Escape" && close()}
    >
      <header className="chat-header">
        <h2>CareBot</h2>
        <button type="button" className="button quiet" onClick={close}>
          Close
        </button>
      </header>

      <div className="chat-messages" role="log" aria-live="polite" tabIndex={0}>
        {messages.map((message, index) => (
          <div key={index} className={`chat-message ${message.role}`}>
            <span className="visually-hidden">
              {message.role === "user" ? "You: " : "CareBot: "}
            </span>
            <p lang={message.lang}>{message.text}</p>
            <TriageDisclaimer text={message.disclaimer} lang={message.lang} />
          </div>
        ))}
        {sending && <p className="chat-typing muted">CareBot is replying…</p>}
        <div ref={endRef} />
      </div>

      {problem && (
        <p className="error chat-error" role="alert">
          {problem}
        </p>
      )}

      <div className="chat-voice">
        <button
          type="button"
          className={`button ${voiceOn ? "danger" : ""}`}
          aria-pressed={voiceOn}
          disabled={voice.phase === "starting"}
          onClick={voiceOn ? stopVoice : startVoice}
        >
          {voiceOn ? "Stop voice" : "Start voice"}
        </button>
        <p className="chat-voice-status muted" role="status">
          {status}
        </p>
      </div>

      <form className="chat-form" onSubmit={send}>
        <label htmlFor="chat-input" className="visually-hidden">
          Message to CareBot
        </label>
        <input
          id="chat-input"
          ref={inputRef}
          value={draft}
          maxLength={MAX_LENGTH}
          autoComplete="off"
          placeholder="Type your message"
          onChange={(event) => setDraft(event.target.value)}
        />
        <button type="submit" className="button primary" disabled={sending || !draft.trim()}>
          Send
        </button>
      </form>
    </section>
  );
}
