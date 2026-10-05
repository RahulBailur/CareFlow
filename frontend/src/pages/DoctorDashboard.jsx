import { useCallback, useEffect, useState } from "react";

import api, { errorMessage } from "../api";
import { useAuth } from "../context/AuthContext";
import { useQueueUpdates } from "../context/QueueSocketContext";
import { formatTime } from "../time";

const STATUS_LABELS = {
  booked: "Waiting",
  in_consultation: "In consultation",
  done: "Done",
  no_show: "No-show",
};

// The buttons a doctor gets for an appointment in each state
const NEXT_ACTIONS = {
  booked: [
    { status: "in_consultation", label: "Start consultation", style: "primary" },
    { status: "no_show", label: "No-show", style: "quiet" },
  ],
  in_consultation: [{ status: "done", label: "Mark done", style: "primary" }],
};

export default function DoctorDashboard() {
  const { user } = useAuth();
  const [queue, setQueue] = useState(null);
  const [error, setError] = useState("");
  const [delay, setDelay] = useState("");
  const [message, setMessage] = useState("");
  const [savedNote, setSavedNote] = useState("");

  const load = useCallback(async () => {
    try {
      setQueue((await api.get(`/api/appointments/queue/${user.id}`)).data);
      setError("");
    } catch (err) {
      setError(errorMessage(err, "Could not load your queue."));
    }
  }, [user.id]);

  useEffect(() => {
    load();
  }, [load]);

  // The doctor's own room receives the full queue on every change
  const { connected } = useQueueUpdates((update) => {
    if (update.entries) setQueue(update);
  });

  async function setStatus(entry, status) {
    try {
      await api.put(`/api/appointments/${entry.appointment_id}/status`, { status });
      await load();
    } catch (err) {
      setError(errorMessage(err, "Could not update the appointment."));
    }
  }

  async function broadcastDelay(minutes) {
    setSavedNote("");
    try {
      await api.post("/api/doctors/status", {
        delay_minutes: minutes,
        message: minutes > 0 ? message.trim() || null : null,
      });
      setDelay("");
      setMessage("");
      setSavedNote(
        minutes > 0 ? "Delay sent to your waiting patients." : "Patients told you are on time.",
      );
      await load();
    } catch (err) {
      setError(errorMessage(err, "Could not send the delay."));
    }
  }

  function handleDelaySubmit(event) {
    event.preventDefault();
    const minutes = Number(delay);
    if (Number.isInteger(minutes) && minutes > 0 && minutes <= 240) broadcastDelay(minutes);
    else setError("Enter a delay between 1 and 240 minutes.");
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h1>Today's queue</h1>
          <p className="muted">
            {user.name} · {user.department}
          </p>
        </div>
        <span className={connected ? "live-dot on" : "live-dot"}>
          {connected ? "Live" : "Reconnecting…"}
        </span>
      </div>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <section className="card">
        <h2>Running late?</h2>
        {queue?.delay_minutes > 0 ? (
          <div className="notice warning">
            Patients currently see a delay of <strong>{queue.delay_minutes} minutes</strong>.
            {queue.delay_message && ` “${queue.delay_message}”`}{" "}
            <button type="button" className="button quiet" onClick={() => broadcastDelay(0)}>
              Clear delay
            </button>
          </div>
        ) : (
          <p className="muted">Patients currently see you as on time.</p>
        )}
        <form className="delay-form" onSubmit={handleDelaySubmit} noValidate>
          <div>
            <label htmlFor="delay">Delay (minutes)</label>
            <input
              id="delay"
              type="number"
              min={1}
              max={240}
              value={delay}
              onChange={(event) => setDelay(event.target.value)}
            />
          </div>
          <div className="grow">
            <label htmlFor="delay-message">Note for patients (optional)</label>
            <input
              id="delay-message"
              maxLength={200}
              value={message}
              onChange={(event) => setMessage(event.target.value)}
            />
          </div>
          <button type="submit" className="button">
            Tell waiting patients
          </button>
        </form>
        {savedNote && (
          <p className="success" role="status">
            {savedNote}
          </p>
        )}
      </section>

      <section>
        <h2>Patients {queue && <span className="muted">· {queue.waiting} waiting</span>}</h2>
        {queue === null && !error && <p className="muted">Loading…</p>}
        {queue?.entries.length === 0 && <p className="muted">No appointments today.</p>}
        <ul className="list">
          {queue?.entries.map((entry) => (
            <li key={entry.appointment_id} className="card row">
              <div>
                <strong>{entry.patient_name}</strong>
                <p className="muted">
                  {formatTime(entry.slot_start)}
                  {entry.reason && ` · ${entry.reason}`}
                </p>
              </div>
              <div className="actions">
                <span className={`status ${entry.status}`}>{STATUS_LABELS[entry.status]}</span>
                {(NEXT_ACTIONS[entry.status] ?? []).map((action) => (
                  <button
                    key={action.status}
                    type="button"
                    className={`button ${action.style}`}
                    onClick={() => setStatus(entry, action.status)}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
