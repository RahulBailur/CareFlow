import { useCallback, useEffect, useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";

import api, { errorMessage } from "../api";
import { addDaysIso, formatDate, formatTime, todayIso } from "../time";

const BOOKING_WINDOW_DAYS = 30;

/** Pick a day, a doctor and a slot. Doubles as the reschedule screen via ?reschedule=<id>. */
export default function BookAppointment() {
  const navigate = useNavigate();
  const [params] = useSearchParams();
  const rescheduleId = params.get("reschedule");
  const fixedDoctor = params.get("doctor");

  const [day, setDay] = useState(todayIso);
  const [department, setDepartment] = useState("");
  const [departments, setDepartments] = useState([]);
  const [doctors, setDoctors] = useState(null);
  const [selected, setSelected] = useState(null); // { doctor, slot }
  const [reason, setReason] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    // Departments only feed the filter, so the page still works if this fails
    api
      .get("/api/hospital/config")
      .then((response) => setDepartments(response.data.departments.map((d) => d.name)))
      .catch(() => setDepartments([]));
  }, []);

  const loadSlots = useCallback(async () => {
    setDoctors(null);
    setSelected(null);
    const query = { date: day };
    if (fixedDoctor) query.doctor_id = fixedDoctor;
    else if (department) query.department = department;
    try {
      setDoctors((await api.get("/api/appointments/availability", { params: query })).data);
    } catch (err) {
      setDoctors([]);
      setError(errorMessage(err, "Could not load available slots."));
    }
  }, [day, department, fixedDoctor]);

  useEffect(() => {
    loadSlots();
  }, [loadSlots]);

  async function confirm() {
    setBusy(true);
    setError("");
    try {
      if (rescheduleId) {
        await api.put(`/api/appointments/${rescheduleId}`, {
          action: "reschedule",
          slot_start: selected.slot.start,
        });
      } else {
        await api.post("/api/appointments/book", {
          doctor_id: selected.doctor.doctor_id,
          slot_start: selected.slot.start,
          reason: reason.trim() || null,
        });
      }
      navigate("/");
    } catch (err) {
      setError(errorMessage(err, "Could not save the appointment."));
      setBusy(false);
      // Someone else may have just taken the slot, so show what is really free now
      if (err.response?.status === 409) loadSlots();
    }
  }

  const withSlots = (doctors ?? []).filter((doctor) => doctor.slots.length > 0);

  return (
    <>
      <h1>{rescheduleId ? "Reschedule appointment" : "Book an appointment"}</h1>

      <div className="card filters">
        <div>
          <label htmlFor="day">Date</label>
          <input
            id="day"
            type="date"
            value={day}
            min={todayIso()}
            max={addDaysIso(todayIso(), BOOKING_WINDOW_DAYS)}
            onChange={(event) => event.target.value && setDay(event.target.value)}
          />
        </div>
        {!fixedDoctor && (
          <div>
            <label htmlFor="department">Department</label>
            <select
              id="department"
              value={department}
              onChange={(event) => setDepartment(event.target.value)}
            >
              <option value="">All departments</option>
              {departments.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </div>
        )}
      </div>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      {doctors === null && <p className="muted">Loading available slots…</p>}
      {doctors !== null && withSlots.length === 0 && (
        <p className="muted">No open slots on {formatDate(day)}. Try another date.</p>
      )}

      {withSlots.map((doctor) => (
        <section key={doctor.doctor_id} className="card">
          <h2>{doctor.doctor_name}</h2>
          <p className="muted">{doctor.department}</p>
          <div className="slots">
            {doctor.slots.map((slot) => {
              const isSelected =
                selected?.doctor.doctor_id === doctor.doctor_id &&
                selected.slot.start === slot.start;
              return (
                <button
                  key={slot.start}
                  type="button"
                  className={isSelected ? "slot selected" : "slot"}
                  aria-pressed={isSelected}
                  onClick={() => setSelected({ doctor, slot })}
                >
                  {formatTime(slot.start)}
                </button>
              );
            })}
          </div>
        </section>
      ))}

      {selected && (
        <section className="card confirm">
          <h2>Confirm</h2>
          <p>
            <strong>{selected.doctor.doctor_name}</strong> · {formatDate(selected.slot.start)} at{" "}
            {formatTime(selected.slot.start)}
          </p>
          {!rescheduleId && (
            <>
              <label htmlFor="reason">Reason for visit (optional)</label>
              <textarea
                id="reason"
                rows={2}
                maxLength={300}
                value={reason}
                onChange={(event) => setReason(event.target.value)}
              />
            </>
          )}
          <button type="button" className="button primary" disabled={busy} onClick={confirm}>
            {busy ? "Saving…" : rescheduleId ? "Move appointment" : "Book appointment"}
          </button>
        </section>
      )}
    </>
  );
}
