import { useEffect, useState } from "react";

import api, { errorMessage } from "../api";
import { formatDate, formatTime } from "../time";

const STATUS_LABELS = {
  booked: "Booked",
  in_consultation: "In consultation",
  done: "Completed",
  no_show: "Missed",
  cancelled: "Cancelled",
};

/** View-only record of the patient's own visits over the last 12 months. */
export default function VisitHistory() {
  const [visits, setVisits] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .get("/api/appointments/me")
      .then((response) => {
        const now = new Date();
        setVisits(
          response.data.filter((a) => a.status !== "booked" || new Date(a.slot_start) < now),
        );
      })
      .catch((err) => setError(errorMessage(err, "Could not load your visit history.")));
  }, []);

  return (
    <>
      <h1>Visit history</h1>
      <p className="muted">Your visits and prescriptions from the last 12 months.</p>

      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {visits === null && !error && <p className="muted">Loading…</p>}
      {visits?.length === 0 && <p className="muted">No past visits yet.</p>}

      {visits?.length > 0 && (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Date</th>
                <th scope="col">Doctor</th>
                <th scope="col">Department</th>
                <th scope="col">Status</th>
                <th scope="col">Prescription</th>
              </tr>
            </thead>
            <tbody>
              {visits.map((visit) => (
                <tr key={visit.id}>
                  <td>
                    {formatDate(visit.slot_start)}
                    <br />
                    <span className="muted">{formatTime(visit.slot_start)}</span>
                  </td>
                  <td>{visit.doctor_name}</td>
                  <td>{visit.department}</td>
                  <td>
                    <span className={`status ${visit.status}`}>{STATUS_LABELS[visit.status]}</span>
                  </td>
                  <td>{visit.prescription ?? <span className="muted">—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
