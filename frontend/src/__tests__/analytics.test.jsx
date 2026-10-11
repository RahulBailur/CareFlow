import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import api from "../api";
import AnalyticsDashboard from "../pages/AnalyticsDashboard";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal()),
  default: { get: vi.fn() },
}));

// jsdom has no layout, so give the charts a size to draw at
vi.mock("recharts", async (importOriginal) => {
  const recharts = await importOriginal();
  const { cloneElement } = await import("react");
  return {
    ...recharts,
    ResponsiveContainer: ({ children }) => cloneElement(children, { width: 600, height: 300 }),
  };
});

const STATS = {
  departments: [
    {
      department: "Cardiology",
      doctors: 2,
      doctors_on_duty: 1,
      appointments_today: 5,
      waiting: 3,
      in_consultation: 1,
      done: 1,
      no_show: 0,
      doctors_running_late: 1,
      longest_delay_minutes: 25,
    },
  ],
  totals: {
    doctors_on_duty: 1,
    appointments_today: 5,
    waiting: 3,
    in_consultation: 1,
    done: 1,
    no_show: 0,
  },
};

const SUMMARY = {
  days: 7,
  turns: 40,
  pipelines: {
    live: { turns: 19, stages: { ttfa_ms: { n: 19, p50: 2290, p95: 3290 } } },
    cascade: {
      turns: 19,
      stages: { stt_ms: { n: 19, p50: 1020, p95: 5110 }, ttfa_ms: { n: 19, p50: 4030, p95: 9000 } },
    },
    text: { turns: 2, stages: { total_ms: { n: 2, p50: 900, p95: 1500 } } },
  },
  intents: { booking: 20, support: 20 },
  answered_by: { "gemini-live": 19, gemini: 9, cache: 6, rules: 6 },
  primary_rate: 0.7,
  cache_rate: 0.15,
  fallback_rate: 0.15,
  llm_errors: { "gemini:429": 4 },
  turns_with_llm_errors: 4,
  blocked: 1,
};

function respond(summary = SUMMARY) {
  api.get.mockImplementation((url) =>
    Promise.resolve({ data: url === "/api/admin/stats" ? STATS : summary }),
  );
}

beforeEach(() => vi.clearAllMocks());

test("shows today's department load", async () => {
  respond();
  render(<AnalyticsDashboard />);

  const row = (await screen.findByRole("cell", { name: "Cardiology" })).closest("tr");
  expect(within(row).getByText("1 of 2")).toBeInTheDocument();
  expect(within(row).getByText("1 (up to 25 min)")).toBeInTheDocument();
});

test("shows the headline rates", async () => {
  respond();
  render(<AnalyticsDashboard />);

  expect(await screen.findByText("Answered by the first-choice model")).toBeInTheDocument();
  expect(screen.getByText("70%")).toBeInTheDocument();
  expect(screen.getByText("Needed a fallback").nextSibling).toHaveTextContent("15%");
  expect(screen.getByText("Replies blocked by guardrails").nextSibling).toHaveTextContent("1");
});

test("every latency figure is also in a table, per pipeline and stage", async () => {
  respond();
  render(<AnalyticsDashboard />);

  const table = (await screen.findByText("Latency per stage and pipeline")).closest("table");
  const live = within(table).getByText("Voice A (real-time)").closest("tr");
  expect(within(live).getByText("Time to first audio")).toBeInTheDocument();
  expect(within(live).getByText("2.29 s")).toBeInTheDocument();
  expect(within(live).getByText("3.29 s")).toBeInTheDocument();
  expect(within(table).getByText("Text chat")).toBeInTheDocument();
});

test("draws the time-to-first-audio chart for the voice pipelines only", async () => {
  respond();
  render(<AnalyticsDashboard />);

  const chart = await screen.findByRole("img", { name: "Time to first audio by voice pipeline" });
  expect(within(chart).getAllByText("Voice A (real-time)").length).toBeGreaterThan(0);
  expect(within(chart).queryByText("Text chat")).not.toBeInTheDocument();
});

test("lists LLM errors by kind", async () => {
  respond();
  render(<AnalyticsDashboard />);

  expect(await screen.findByText("gemini:429")).toBeInTheDocument();
});

test("changing the period asks for that window", async () => {
  respond();
  const user = userEvent.setup();
  render(<AnalyticsDashboard />);
  await screen.findByText("Turns with LLM errors");

  await user.selectOptions(screen.getByLabelText("Period"), "30");

  expect(api.get).toHaveBeenLastCalledWith("/api/admin/analytics", { params: { days: 30 } });
});

test("an empty period says so instead of drawing empty charts", async () => {
  respond({ ...SUMMARY, turns: 0, pipelines: {}, intents: {}, answered_by: {}, llm_errors: {} });
  render(<AnalyticsDashboard />);

  expect(
    await screen.findByText("No CareBot conversations in this period yet."),
  ).toBeInTheDocument();
  expect(screen.queryByRole("img")).not.toBeInTheDocument();
});

test("a failure is explained", async () => {
  api.get.mockRejectedValue({ response: { status: 403, data: { detail: "Not permitted" } } });
  render(<AnalyticsDashboard />);

  expect(await screen.findByRole("alert")).toHaveTextContent("Not permitted");
});

test("stages are listed in the order a turn goes through them, Voice A first", async () => {
  respond();
  render(<AnalyticsDashboard />);

  const table = (await screen.findByText("Latency per stage and pipeline")).closest("table");
  const rows = within(table)
    .getAllByRole("row")
    .slice(1)
    .map((row) => [...row.cells].slice(0, 2).map((cell) => cell.textContent));
  expect(rows).toEqual([
    ["Voice A (real-time)", "Time to first audio"],
    ["Voice B (cascaded)", "Speech to text"],
    ["Voice B (cascaded)", "Time to first audio"],
    ["Text chat", "Whole turn"],
  ]);
});

test("turns from before the tier was recorded are named, and said to be left out", async () => {
  respond({
    ...SUMMARY,
    answered_by: { ...SUMMARY.answered_by, unknown: 3 },
    turns_without_tier: 3,
  });
  render(<AnalyticsDashboard />);

  expect(await screen.findByText("3 older ones left out of the rates")).toBeInTheDocument();
});
