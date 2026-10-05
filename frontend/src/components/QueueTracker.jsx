import { useEffect, useState } from "react";

import api, { errorMessage } from "../api";
import { useQueueUpdates } from "../context/QueueSocketContext";
import { formatTime } from "../time";

function headline(view) {
  if (view.status === "in_consultation") return "It's your turn — the doctor is seeing you now";
  if (view.patients_ahead === 0) return "You're next";
  if (view.patients_ahead === 1) return "1 patient ahead of you";
  return `${view.patients_ahead} patients ahead of you`;
}

/** Live queue position for one of today's appointments. */
export default function QueueTracker({ appointment }) {
  const [view, setView] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    api
      .get(`/api/appointments/queue/${appointment.doctor_id}`)
      .then((response) => !cancelled && setView(response.data))
      .catch((err) => !cancelled && setError(errorMessage(err, "Could not load the queue.")));
    return () => {
      cancelled = true;
    };
  }, [appointment.doctor_id]);

  const { connected } = useQueueUpdates((update) => {
    if (update.appointment_id === appointment.id) setView(update);
  });

  return (
    <section className="card queue-tracker" aria-live="polite">
      <header className="card-header">
        <div>
          <h3>{appointment.doctor_name}</h3>
          <p className="muted">
            {appointment.department} · booked for {formatTime(appointment.slot_start)}
          </p>
        </div>
        <span className={connected ? "live-dot on" : "live-dot"}>
          {connected ? "Live" : "Reconnecting…"}
        </span>
      </header>

      {error && <p className="error">{error}</p>}
      {!view && !error && <p className="muted">Loading your place in the queue…</p>}

      {view?.in_queue && (
        <>
          <p className="queue-position">{headline(view)}</p>
          {view.delay_minutes > 0 ? (
            <div className="notice warning">
              <strong>Running about {view.delay_minutes} minutes late.</strong> {view.delay_message}{" "}
              Expected around {formatTime(view.eta)}.
            </div>
          ) : (
            <p className="muted">Running on time.</p>
          )}
        </>
      )}
      {view && !view.in_queue && (
        <p className="muted">This appointment is {view.status.replace("_", " ")}.</p>
      )}
    </section>
  );
}
