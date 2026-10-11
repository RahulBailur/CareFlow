import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import api, { errorMessage } from "../api";

export const PIPELINE_NAMES = {
  live: "Voice A (real-time)",
  cascade: "Voice B (cascaded)",
  text: "Text chat",
};

export const STAGE_NAMES = {
  end_of_speech_ms: "End-of-speech detection",
  stt_ms: "Speech to text",
  routing_ms: "Routing",
  llm_ms: "LLM",
  tts_first_chunk_ms: "TTS first chunk",
  ttfa_ms: "Time to first audio",
  total_ms: "Whole turn",
};

const TIER_NAMES = {
  "gemini-live": "Gemini Live",
  gemini: "Gemini",
  ollama: "Ollama (local)",
  cache: "Response cache",
  rules: "Rule-based reply",
  unknown: "Not recorded (older turns)",
};

const PIPELINE_ORDER = ["live", "cascade", "text"];
const STAGE_ORDER = Object.keys(STAGE_NAMES);
const byOrder = (order) => (a, b) => (order.indexOf(a) + 1 || 99) - (order.indexOf(b) + 1 || 99);

const WINDOWS = [
  { days: 1, label: "Last 24 hours" },
  { days: 7, label: "Last 7 days" },
  { days: 30, label: "Last 30 days" },
];

const seconds = (ms) => `${(ms / 1000).toFixed(ms < 10000 ? 2 : 1)} s`;
const percent = (share) => `${Math.round(share * 100)}%`;

function Stat({ label, value, note }) {
  return (
    <div className="stat">
      <p className="stat-label">{label}</p>
      <p className="stat-value">{value}</p>
      {note && <p className="stat-note">{note}</p>}
    </div>
  );
}

/** Recharts' tooltip, in the page's own colours. */
const tooltipProps = {
  cursor: { fill: "var(--accent-soft)" },
  contentStyle: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    color: "var(--text)",
  },
  labelStyle: { color: "var(--text)", fontWeight: 600 },
  itemStyle: { color: "var(--text)" },
};

const axisProps = {
  stroke: "var(--border)",
  tick: { fill: "var(--muted)", fontSize: 12 },
  tickLine: false,
};

function LatencyChart({ pipelines }) {
  const rows = ["live", "cascade"]
    .filter((name) => pipelines[name]?.stages?.ttfa_ms)
    .map((name) => ({
      name: PIPELINE_NAMES[name],
      p50: pipelines[name].stages.ttfa_ms.p50 / 1000,
      p95: pipelines[name].stages.ttfa_ms.p95 / 1000,
    }));
  if (rows.length === 0) return <p className="muted">No voice turns in this period yet.</p>;
  return (
    <div className="chart" role="img" aria-label="Time to first audio by voice pipeline">
      <ResponsiveContainer width="100%" height={60 + rows.length * 70}>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 56 }} barGap={2}>
          <CartesianGrid horizontal={false} stroke="var(--border)" />
          <XAxis type="number" unit=" s" {...axisProps} />
          <YAxis type="category" dataKey="name" width={150} {...axisProps} />
          <Tooltip {...tooltipProps} formatter={(value) => `${value.toFixed(2)} s`} />
          {/* The swatch carries the colour; the words stay in text colour */}
          <Legend
            wrapperStyle={{ fontSize: 13 }}
            formatter={(value) => <span style={{ color: "var(--muted)" }}>{value}</span>}
          />
          {[
            ["p50", "Median (P50)", "var(--series-1)"],
            ["p95", "Slow turns (P95)", "var(--series-2)"],
          ].map(([key, name, colour]) => (
            <Bar
              key={key}
              dataKey={key}
              name={name}
              fill={colour}
              barSize={16}
              radius={[0, 4, 4, 0]}
            >
              <LabelList
                dataKey={key}
                position="right"
                formatter={(value) => `${value.toFixed(1)} s`}
                style={{ fill: "var(--muted)", fontSize: 12 }}
              />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** One measure, a few categories: a single-series bar chart with the share at each tip. */
function CountChart({ label, counts, names = {} }) {
  const total = Object.values(counts).reduce((sum, n) => sum + n, 0);
  const rows = Object.entries(counts).map(([key, count]) => ({
    name: names[key] ?? key,
    count,
    share: total ? count / total : 0,
  }));
  if (rows.length === 0) return <p className="muted">Nothing yet.</p>;
  return (
    <div className="chart" role="img" aria-label={label}>
      <ResponsiveContainer width="100%" height={30 + rows.length * 36}>
        <BarChart data={rows} layout="vertical" margin={{ left: 8, right: 48 }}>
          <CartesianGrid horizontal={false} stroke="var(--border)" />
          <XAxis type="number" allowDecimals={false} {...axisProps} />
          <YAxis type="category" dataKey="name" width={140} {...axisProps} />
          <Tooltip
            {...tooltipProps}
            formatter={(value, _, item) => [`${value} (${percent(item.payload.share)})`, "Turns"]}
          />
          <Bar dataKey="count" fill="var(--series-1)" barSize={16} radius={[0, 4, 4, 0]}>
            <LabelList
              dataKey="share"
              position="right"
              formatter={percent}
              style={{ fill: "var(--muted)", fontSize: 12 }}
            />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function StageTable({ pipelines }) {
  const names = Object.keys(pipelines).sort(byOrder(PIPELINE_ORDER));
  return (
    <div className="table-wrap">
      <table>
        <caption className="visually-hidden">Latency per stage and pipeline</caption>
        <thead>
          <tr>
            <th scope="col">Pipeline</th>
            <th scope="col">Stage</th>
            <th scope="col">P50</th>
            <th scope="col">P95</th>
            <th scope="col">Turns</th>
          </tr>
        </thead>
        <tbody>
          {names.flatMap((pipeline) =>
            Object.entries(pipelines[pipeline].stages)
              .sort(([a], [b]) => byOrder(STAGE_ORDER)(a, b))
              .map(([stage, stats]) => (
                <tr key={`${pipeline}-${stage}`}>
                  <td>{PIPELINE_NAMES[pipeline] ?? pipeline}</td>
                  <td>{STAGE_NAMES[stage] ?? stage}</td>
                  <td>{seconds(stats.p50)}</td>
                  <td>{seconds(stats.p95)}</td>
                  <td>{stats.n}</td>
                </tr>
              )),
          )}
        </tbody>
      </table>
    </div>
  );
}

function DepartmentTable({ departments }) {
  return (
    <div className="table-wrap">
      <table>
        <caption className="visually-hidden">Department load today</caption>
        <thead>
          <tr>
            <th scope="col">Department</th>
            <th scope="col">Doctors on duty</th>
            <th scope="col">Appointments</th>
            <th scope="col">Waiting</th>
            <th scope="col">In consultation</th>
            <th scope="col">Done</th>
            <th scope="col">No-show</th>
            <th scope="col">Running late</th>
          </tr>
        </thead>
        <tbody>
          {departments.map((row) => (
            <tr key={row.department}>
              <td>{row.department}</td>
              <td>
                {row.doctors_on_duty} of {row.doctors}
              </td>
              <td>{row.appointments_today}</td>
              <td>{row.waiting}</td>
              <td>{row.in_consultation}</td>
              <td>{row.done}</td>
              <td>{row.no_show}</td>
              <td>
                {row.doctors_running_late
                  ? `${row.doctors_running_late} (up to ${row.longest_delay_minutes} min)`
                  : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Admin: department load today, and CareBot's latency, answers and fallbacks. */
export default function AnalyticsDashboard() {
  const [days, setDays] = useState(7);
  const [stats, setStats] = useState(null);
  const [carebot, setCarebot] = useState(null);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    try {
      const [load, summary] = await Promise.all([
        api.get("/api/admin/stats"),
        api.get("/api/admin/analytics", { params: { days } }),
      ]);
      setStats(load.data);
      setCarebot(summary.data);
      setError("");
    } catch (err) {
      setError(errorMessage(err, "Could not load the analytics."));
    }
  }, [days]);

  useEffect(() => {
    load();
  }, [load]);

  return (
    <>
      <div className="page-header">
        <h1>Overview</h1>
        <button type="button" className="button" onClick={load}>
          Refresh
        </button>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}

      <section aria-labelledby="today-heading">
        <h2 id="today-heading">Today</h2>
        {!stats && !error && <p className="muted">Loading…</p>}
        {stats && (
          <>
            <div className="stats">
              <Stat label="Appointments" value={stats.totals.appointments_today} />
              <Stat label="Waiting" value={stats.totals.waiting} />
              <Stat label="In consultation" value={stats.totals.in_consultation} />
              <Stat label="Doctors on duty" value={stats.totals.doctors_on_duty} />
            </div>
            <DepartmentTable departments={stats.departments} />
          </>
        )}
      </section>

      <section aria-labelledby="carebot-heading">
        <div className="page-header section-header">
          <h2 id="carebot-heading">CareBot</h2>
          <label className="inline-label">
            <span className="visually-hidden">Period</span>
            <select value={days} onChange={(event) => setDays(Number(event.target.value))}>
              {WINDOWS.map((window) => (
                <option key={window.days} value={window.days}>
                  {window.label}
                </option>
              ))}
            </select>
          </label>
        </div>

        {carebot && carebot.turns === 0 && (
          <p className="muted">No CareBot conversations in this period yet.</p>
        )}
        {carebot && carebot.turns > 0 && (
          <>
            <div className="stats">
              <Stat
                label="Turns"
                value={carebot.turns}
                note={
                  carebot.turns_without_tier
                    ? `${carebot.turns_without_tier} older ones left out of the rates`
                    : undefined
                }
              />
              <Stat
                label="Answered by the first-choice model"
                value={percent(carebot.primary_rate)}
              />
              <Stat label="From the response cache" value={percent(carebot.cache_rate)} />
              <Stat
                label="Needed a fallback"
                value={percent(carebot.fallback_rate)}
                note="Ollama or a rule-based reply"
              />
              <Stat label="Turns with LLM errors" value={carebot.turns_with_llm_errors} />
              <Stat
                label="Replies blocked by guardrails"
                value={carebot.blocked}
                note="Diagnosis, prescription or dosage"
              />
            </div>

            <h3>Time to first audio</h3>
            <p className="muted">
              From the moment the patient stops speaking to the first sound of the reply.
            </p>
            <LatencyChart pipelines={carebot.pipelines} />

            <h3>Latency by stage</h3>
            <StageTable pipelines={carebot.pipelines} />

            <div className="chart-grid">
              <div>
                <h3>Who answered</h3>
                <CountChart
                  label="Which tier answered each turn"
                  counts={carebot.answered_by}
                  names={TIER_NAMES}
                />
              </div>
              <div>
                <h3>What patients asked about</h3>
                <CountChart label="Intent of each turn" counts={carebot.intents} />
              </div>
            </div>

            <h3>LLM errors</h3>
            {Object.keys(carebot.llm_errors).length === 0 ? (
              <p className="muted">None in this period.</p>
            ) : (
              <ul className="list error-list">
                {Object.entries(carebot.llm_errors).map(([kind, count]) => (
                  <li key={kind}>
                    <code>{kind}</code> — {count}
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </section>
    </>
  );
}
