import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import api from "../api";
import BookAppointment from "../pages/BookAppointment";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal()),
  default: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
}));

const SLOT = { start: "2026-10-06T03:30:00Z", end: "2026-10-06T03:45:00Z" }; // 9:00 am IST
const LATER = { start: "2026-10-06T03:45:00Z", end: "2026-10-06T04:00:00Z" }; // 9:15 am IST
const AVAILABILITY = [
  { doctor_id: "d1", doctor_name: "Dr. Asha Rao", department: "Cardiology", slots: [SLOT, LATER] },
  { doctor_id: "d2", doctor_name: "Dr. Fully Booked", department: "ENT", slots: [] },
];

function mockApi(availability = AVAILABILITY) {
  api.get.mockImplementation((url) =>
    Promise.resolve({
      data:
        url === "/api/hospital/config" ? { departments: [{ name: "Cardiology" }] } : availability,
    }),
  );
}

function renderAt(path = "/book") {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <Routes>
        <Route path="/book" element={<BookAppointment />} />
        <Route path="/" element={<p>Home page</p>} />
      </Routes>
    </MemoryRouter>,
  );
}

beforeEach(() => vi.clearAllMocks());

test("shows open slots in hospital time and hides doctors with none", async () => {
  mockApi();

  renderAt();

  expect(await screen.findByRole("button", { name: /9:00\s?am/i })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /9:15\s?am/i })).toBeInTheDocument();
  expect(screen.getByText("Dr. Asha Rao")).toBeInTheDocument();
  expect(screen.queryByText("Dr. Fully Booked")).not.toBeInTheDocument();
});

test("books the chosen slot and returns home", async () => {
  mockApi();
  api.post.mockResolvedValue({ data: {} });
  const user = userEvent.setup();
  renderAt();

  await user.click(await screen.findByRole("button", { name: /9:15\s?am/i }));
  await user.type(screen.getByLabelText("Reason for visit (optional)"), "Chest pain");
  await user.click(screen.getByRole("button", { name: "Book appointment" }));

  expect(api.post).toHaveBeenCalledWith("/api/appointments/book", {
    doctor_id: "d1",
    slot_start: LATER.start,
    reason: "Chest pain",
  });
  expect(await screen.findByText("Home page")).toBeInTheDocument();
});

test("when the slot was just taken, says so and reloads what is free", async () => {
  mockApi();
  api.post.mockRejectedValue({
    response: { status: 409, data: { detail: "That slot has just been taken" } },
  });
  const user = userEvent.setup();
  renderAt();
  await user.click(await screen.findByRole("button", { name: /9:00\s?am/i }));
  const loadsBefore = api.get.mock.calls.length;

  await user.click(screen.getByRole("button", { name: "Book appointment" }));

  expect(await screen.findByRole("alert")).toHaveTextContent("That slot has just been taken");
  expect(api.get.mock.calls.length).toBeGreaterThan(loadsBefore);
  expect(screen.queryByText("Home page")).not.toBeInTheDocument();
});

test("in reschedule mode it moves the existing appointment instead of booking", async () => {
  mockApi();
  api.put.mockResolvedValue({ data: {} });
  const user = userEvent.setup();
  renderAt("/book?reschedule=a1&doctor=d1");

  expect(screen.getByRole("heading", { name: "Reschedule appointment" })).toBeInTheDocument();
  expect(screen.queryByLabelText("Department")).not.toBeInTheDocument();
  await user.click(await screen.findByRole("button", { name: /9:15\s?am/i }));
  await user.click(screen.getByRole("button", { name: "Move appointment" }));

  expect(api.put).toHaveBeenCalledWith("/api/appointments/a1", {
    action: "reschedule",
    slot_start: LATER.start,
  });
  expect(api.post).not.toHaveBeenCalled();
});

test("says so when the day has no open slots", async () => {
  mockApi([]);

  renderAt();

  expect(await screen.findByText(/No open slots on/)).toBeInTheDocument();
});
