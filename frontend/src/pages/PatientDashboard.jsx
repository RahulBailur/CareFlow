import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import api, { errorMessage } from "../api";
import { APPOINTMENTS_CHANGED } from "../components/ChatBot";
import QueueTracker from "../components/QueueTracker";
import { useAuth } from "../context/AuthContext";
import { useQueueUpdates } from "../context/QueueSocketContext";
import { formatDate, formatTime, isToday } from "../time";

const IN_QUEUE = ["booked", "in_consultation"];

export default function PatientDashboard() {
  const { user } = useAuth();
  const [appointments, setAppointments] = useState(null);
  const [error, setError] = useState("");
  const [confirmingCancel, setConfirmingCancel] = useState(null);

  const load = useCallback(async () => {
    try {
      setAppointments((await api.get("/api/appointments/me")).data);
      setError("");
    } catch (err) {
      setError(errorMessage(err, "Could not load your appointments."));
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // CareBot booked, moved or cancelled something
  useEffect(() => {
    window.addEventListener(APPOINTMENTS_CHANGED, load);
    return () => window.removeEventListener(APPOINTMENTS_CHANGED, load);
  }, [load]);

  // A finished or cancelled appointment should leave the lists without a manual refresh
  useQueueUpdates((update) => {
    if (update.appointment_id && !update.in_queue) load();
  });

  async function cancel(appointment) {
    try {
      await api.put(`/api/appointments/${appointment.id}`, { action: "cancel" });
      setConfirmingCancel(null);
      await load();
    } catch (err) {
      setError(errorMessage(err, "Could not cancel the appointment."));
    }
  }

  const all = appointments ?? [];
  const today = all.filter((a) => IN_QUEUE.includes(a.status) && isToday(a.slot_start));
  const upcoming = all
    .filter((a) => a.status === "booked" && new Date(a.slot_start) > new Date())
    .sort((a, b) => new Date(a.slot_start) - new Date(b.slot_start));

  return (
    <>
      <div className="page-header">
        <h1>Hello, {user.name.split(" ")[0]}</h1>
        <Link to="/book" className="button primary">
          Book an appointment
        </Link>
      </div>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {today.length > 0 && (
        <section>
          <h2>Today's queue</h2>
          {today.map((appointment) => (
            <QueueTracker key={appointment.id} appointment={appointment} />
          ))}
        </section>
      )}

      <section>
        <h2>Upcoming appointments</h2>
        {appointments === null && !error && <p className="muted">Loading…</p>}
        {appointments !== null && upcoming.length === 0 && (
          <p className="muted">
            You have no upcoming appointments. <Link to="/book">Book one</Link>.
          </p>
        )}
        <ul className="list">
          {upcoming.map((appointment) => (
            <li key={appointment.id} className="card row">
              <div>
                <strong>{appointment.doctor_name}</strong>
                <p className="muted">
                  {appointment.department} · {formatDate(appointment.slot_start)} at{" "}
                  {formatTime(appointment.slot_start)}
                </p>
              </div>
              {confirmingCancel === appointment.id ? (
                <div className="actions">
                  <span>Cancel this appointment?</span>
                  <button
                    type="button"
                    className="button danger"
                    onClick={() => cancel(appointment)}
                  >
                    Yes, cancel
                  </button>
                  <button
                    type="button"
                    className="button quiet"
                    onClick={() => setConfirmingCancel(null)}
                  >
                    Keep it
                  </button>
                </div>
              ) : (
                <div className="actions">
                  <Link
                    className="button"
                    to={`/book?reschedule=${appointment.id}&doctor=${appointment.doctor_id}`}
                  >
                    Reschedule
                  </Link>
                  <button
                    type="button"
                    className="button quiet"
                    onClick={() => setConfirmingCancel(appointment.id)}
                  >
                    Cancel
                  </button>
                </div>
              )}
            </li>
          ))}
        </ul>
      </section>
    </>
  );
}
