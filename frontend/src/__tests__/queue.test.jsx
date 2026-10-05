import { act, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import api from "../api";
import QueueTracker from "../components/QueueTracker";
import DoctorDashboard from "../pages/DoctorDashboard";

const socket = vi.hoisted(() => ({ emit: null }));

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal()),
  default: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
}));

// Stand in for the live socket: tests push updates by calling socket.emit(...)
vi.mock("../context/QueueSocketContext", () => ({
  useQueueUpdates: (handler) => {
    socket.emit = handler;
    return { connected: true };
  },
}));

vi.mock("../context/AuthContext", () => ({
  useAuth: () => ({ user: { id: "d1", name: "Dr. Asha Rao", department: "Cardiology" } }),
}));

beforeEach(() => vi.clearAllMocks());

describe("QueueTracker", () => {
  const appointment = {
    id: "a1",
    doctor_id: "d1",
    doctor_name: "Dr. Asha Rao",
    department: "Cardiology",
    slot_start: "2026-10-06T04:00:00Z",
  };
  const view = {
    appointment_id: "a1",
    status: "booked",
    in_queue: true,
    patients_ahead: 3,
    delay_minutes: 0,
    delay_message: null,
    eta: "2026-10-06T04:00:00Z",
  };

  test("shows how many patients are ahead", async () => {
    api.get.mockResolvedValue({ data: view });

    render(<QueueTracker appointment={appointment} />);

    expect(await screen.findByText("3 patients ahead of you")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledWith("/api/appointments/queue/d1");
    expect(screen.getByText("Running on time.")).toBeInTheDocument();
  });

  test("moves up live when the server pushes an update", async () => {
    api.get.mockResolvedValue({ data: view });
    render(<QueueTracker appointment={appointment} />);
    await screen.findByText("3 patients ahead of you");

    act(() => socket.emit({ ...view, patients_ahead: 1 }));
    expect(screen.getByText("1 patient ahead of you")).toBeInTheDocument();

    act(() => socket.emit({ ...view, patients_ahead: 0 }));
    expect(screen.getByText("You're next")).toBeInTheDocument();
  });

  test("ignores updates meant for a different appointment", async () => {
    api.get.mockResolvedValue({ data: view });
    render(<QueueTracker appointment={appointment} />);
    await screen.findByText("3 patients ahead of you");

    act(() => socket.emit({ ...view, appointment_id: "someone-else", patients_ahead: 0 }));

    expect(screen.getByText("3 patients ahead of you")).toBeInTheDocument();
  });

  test("shows a delay with the new expected time", async () => {
    api.get.mockResolvedValue({ data: view });
    render(<QueueTracker appointment={appointment} />);
    await screen.findByText("3 patients ahead of you");

    act(() =>
      socket.emit({
        ...view,
        delay_minutes: 20,
        delay_message: "Emergency case.",
        eta: "2026-10-06T04:20:00Z",
      }),
    );

    const notice = screen.getByText(/Running about 20 minutes late/).closest(".notice");
    expect(notice).toHaveTextContent("Emergency case.");
    expect(notice).toHaveTextContent(/Expected around 9:50\s?am/i); // 04:20 UTC in IST
  });

  test("says so when it is the patient's turn", async () => {
    api.get.mockResolvedValue({ data: { ...view, status: "in_consultation", patients_ahead: 0 } });

    render(<QueueTracker appointment={appointment} />);

    expect(await screen.findByText(/It's your turn/)).toBeInTheDocument();
  });
});

describe("DoctorDashboard", () => {
  const queue = {
    doctor_id: "d1",
    delay_minutes: 0,
    delay_message: null,
    waiting: 1,
    entries: [
      {
        appointment_id: "a1",
        patient_name: "Kiran Gowda",
        slot_start: "2026-10-06T03:30:00Z",
        status: "in_consultation",
        reason: null,
      },
      {
        appointment_id: "a2",
        patient_name: "Priya Menon",
        slot_start: "2026-10-06T03:45:00Z",
        status: "booked",
        reason: "Follow-up",
      },
    ],
  };

  const rowFor = (name) => screen.getByText(name).closest("li");

  test("lists today's patients with the right actions for each", async () => {
    api.get.mockResolvedValue({ data: queue });

    render(<DoctorDashboard />);

    expect(await screen.findByText("Kiran Gowda")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledWith("/api/appointments/queue/d1");
    expect(within(rowFor("Kiran Gowda")).getByRole("button", { name: "Mark done" })).toBeVisible();
    const waiting = within(rowFor("Priya Menon"));
    expect(waiting.getByRole("button", { name: "Start consultation" })).toBeVisible();
    expect(waiting.getByRole("button", { name: "No-show" })).toBeVisible();
    expect(waiting.getByText(/Follow-up/)).toBeInTheDocument();
  });

  test("starting a consultation updates that appointment", async () => {
    api.get.mockResolvedValue({ data: queue });
    api.put.mockResolvedValue({ data: {} });
    render(<DoctorDashboard />);
    await screen.findByText("Priya Menon");

    await userEvent
      .setup()
      .click(within(rowFor("Priya Menon")).getByRole("button", { name: "Start consultation" }));

    expect(api.put).toHaveBeenCalledWith("/api/appointments/a2/status", {
      status: "in_consultation",
    });
  });

  test("broadcasts a delay with its note", async () => {
    api.get.mockResolvedValue({ data: queue });
    api.post.mockResolvedValue({ data: {} });
    const user = userEvent.setup();
    render(<DoctorDashboard />);
    await screen.findByText("Priya Menon");

    await user.type(screen.getByLabelText("Delay (minutes)"), "25");
    await user.type(screen.getByLabelText("Note for patients (optional)"), "Emergency case");
    await user.click(screen.getByRole("button", { name: "Tell waiting patients" }));

    expect(api.post).toHaveBeenCalledWith("/api/doctors/status", {
      delay_minutes: 25,
      message: "Emergency case",
    });
    expect(await screen.findByRole("status")).toHaveTextContent("Delay sent");
  });

  test("refuses a delay outside the allowed range", async () => {
    api.get.mockResolvedValue({ data: queue });
    const user = userEvent.setup();
    render(<DoctorDashboard />);
    await screen.findByText("Priya Menon");

    await user.type(screen.getByLabelText("Delay (minutes)"), "900");
    await user.click(screen.getByRole("button", { name: "Tell waiting patients" }));

    expect(screen.getByRole("alert")).toHaveTextContent("between 1 and 240 minutes");
    expect(api.post).not.toHaveBeenCalled();
  });

  test("replaces the list when the server pushes the queue", async () => {
    api.get.mockResolvedValue({ data: queue });
    render(<DoctorDashboard />);
    await screen.findByText("Priya Menon");

    act(() => socket.emit({ ...queue, waiting: 0, entries: [queue.entries[0]] }));

    expect(screen.queryByText("Priya Menon")).not.toBeInTheDocument();
    expect(screen.getByText("Kiran Gowda")).toBeInTheDocument();
  });
});
