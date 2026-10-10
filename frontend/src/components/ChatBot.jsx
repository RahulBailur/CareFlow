import { useEffect, useRef, useState } from "react";

import api, { errorMessage } from "../api";
import { useAuth } from "../context/AuthContext";
import TriageDisclaimer from "./TriageDisclaimer";

const MAX_LENGTH = 500;
const CHANGING_TOOLS = ["book_slot", "cancel_appointment", "reschedule_appointment"];
export const APPOINTMENTS_CHANGED = "careflow:appointments-changed";

const WELCOME = {
  role: "bot",
  text: "Hi, I'm CareBot. I can book, change or cancel an appointment, suggest which department to visit, show your visit history, and answer questions about the hospital. You can write in English, Hindi or Kannada.",
};

/** CareBot text chat for patients: a floating button that opens a conversation panel. */
export default function ChatBot() {
  const { user } = useAuth();
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([WELCOME]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const sessionId = useRef(null);
  const inputRef = useRef(null);
  const launcherRef = useRef(null);
  const endRef = useRef(null);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end" });
  }, [messages, sending, open]);

  if (user?.role !== "patient") return null;

  function close() {
    setOpen(false);
    // After the panel unmounts, so focus has somewhere to land
    setTimeout(() => launcherRef.current?.focus(), 0);
  }

  async function send(event) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sending) return;
    setMessages((current) => [...current, { role: "user", text }]);
    setDraft("");
    setError("");
    setSending(true);
    try {
      const { data } = await api.post("/api/chat", {
        message: text,
        ...(sessionId.current && { session_id: sessionId.current }),
      });
      sessionId.current = data.session_id;
      setMessages((current) => [
        ...current,
        { role: "bot", text: data.reply, lang: data.language, disclaimer: data.disclaimer },
      ]);
      if (data.tools_called?.some((tool) => CHANGING_TOOLS.includes(tool))) {
        window.dispatchEvent(new Event(APPOINTMENTS_CHANGED));
      }
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

      {error && (
        <p className="error chat-error" role="alert">
          {error}
        </p>
      )}

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
