import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import api from "../api";
import ChatBot, { APPOINTMENTS_CHANGED } from "../components/ChatBot";
import TriageDisclaimer from "../components/TriageDisclaimer";

const auth = vi.hoisted(() => ({ user: null }));

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal()),
  default: { post: vi.fn() },
}));

vi.mock("../context/AuthContext", () => ({ useAuth: () => ({ user: auth.user }) }));

const PATIENT = { id: "p1", name: "Ananya Sharma", role: "patient" };

function reply(overrides = {}) {
  return {
    data: {
      reply: "OPD timings are 9 am to 1 pm.",
      session_id: "session-abc-123",
      intent: "support",
      language: "en",
      disclaimer: null,
      blocked: false,
      tools_called: [],
      ...overrides,
    },
  };
}

async function openChat() {
  const user = userEvent.setup();
  render(<ChatBot />);
  await user.click(screen.getByRole("button", { name: "Ask CareBot" }));
  return user;
}

async function say(user, text) {
  await user.type(screen.getByLabelText("Message to CareBot"), text);
  await user.click(screen.getByRole("button", { name: "Send" }));
}

beforeEach(() => {
  vi.clearAllMocks();
  auth.user = PATIENT;
});

test.each([null, { role: "doctor" }, { role: "admin" }])(
  "is not offered when the user is %o",
  (user) => {
    auth.user = user;
    const { container } = render(<ChatBot />);

    expect(container).toBeEmptyDOMElement();
  },
);

test("opens from the launcher with a welcome message and the input focused", async () => {
  await openChat();

  const log = screen.getByRole("log");
  expect(within(log).getByText(/I'm CareBot/)).toBeInTheDocument();
  expect(screen.getByLabelText("Message to CareBot")).toHaveFocus();
  expect(api.post).not.toHaveBeenCalled();
});

test("closes with the Close button or Escape and returns focus to the launcher", async () => {
  const user = await openChat();

  await user.click(screen.getByRole("button", { name: "Close" }));
  const launcher = await screen.findByRole("button", { name: "Ask CareBot" });
  await vi.waitFor(() => expect(launcher).toHaveFocus());

  await user.click(launcher);
  await user.keyboard("{Escape}");
  expect(screen.queryByRole("log")).not.toBeInTheDocument();
});

test("sends a message and shows the reply", async () => {
  api.post.mockResolvedValue(reply());
  const user = await openChat();

  await say(user, "What are the OPD timings?");

  expect(api.post).toHaveBeenCalledWith("/api/chat", { message: "What are the OPD timings?" });
  const log = screen.getByRole("log");
  expect(within(log).getByText("What are the OPD timings?")).toBeInTheDocument();
  expect(await within(log).findByText("OPD timings are 9 am to 1 pm.")).toBeInTheDocument();
  expect(screen.getByLabelText("Message to CareBot")).toHaveValue("");
});

test("Enter sends the message", async () => {
  api.post.mockResolvedValue(reply());
  const user = await openChat();

  await user.type(screen.getByLabelText("Message to CareBot"), "hello{Enter}");

  expect(api.post).toHaveBeenCalledTimes(1);
});

test("continues the same conversation by sending the session id back", async () => {
  api.post.mockResolvedValue(reply());
  const user = await openChat();

  await say(user, "first");
  await screen.findByText("OPD timings are 9 am to 1 pm.");
  await say(user, "second");

  expect(api.post).toHaveBeenLastCalledWith("/api/chat", {
    message: "second",
    session_id: "session-abc-123",
  });
});

test("a triage reply shows the disclaimer the server sent, in its language", async () => {
  api.post.mockResolvedValue(
    reply({
      reply: "जनरल मेडिसिन विभाग में जाइए।",
      intent: "triage",
      language: "hi",
      disclaimer: "यह कोई निदान नहीं है। कृपया डॉक्टर से सलाह लें।",
    }),
  );
  const user = await openChat();

  await say(user, "मुझे बुखार है");

  const note = await screen.findByRole("note");
  expect(note).toHaveTextContent("यह कोई निदान नहीं है। कृपया डॉक्टर से सलाह लें।");
  expect(note).toHaveAttribute("lang", "hi");
  expect(screen.getByText("जनरल मेडिसिन विभाग में जाइए।")).toHaveAttribute("lang", "hi");
});

test("replies that are not about symptoms show no disclaimer", async () => {
  api.post.mockResolvedValue(reply());
  const user = await openChat();

  await say(user, "timings?");
  await screen.findByText("OPD timings are 9 am to 1 pm.");

  expect(screen.queryByRole("note")).not.toBeInTheDocument();
});

test("the disclaimer component renders nothing without text", () => {
  const { container } = render(<TriageDisclaimer text={null} />);

  expect(container).toBeEmptyDOMElement();
});

test("blank messages are not sent", async () => {
  const user = await openChat();

  expect(screen.getByRole("button", { name: "Send" })).toBeDisabled();
  await user.type(screen.getByLabelText("Message to CareBot"), "   {Enter}");

  expect(api.post).not.toHaveBeenCalled();
});

test("while CareBot is replying, a second message cannot be sent", async () => {
  let finish;
  api.post.mockReturnValue(new Promise((resolve) => (finish = resolve)));
  const user = await openChat();

  await say(user, "hello");
  expect(screen.getByText("CareBot is replying…")).toBeInTheDocument();
  await user.type(screen.getByLabelText("Message to CareBot"), "again{Enter}");
  expect(api.post).toHaveBeenCalledTimes(1);

  finish(reply());
  expect(await screen.findByText("OPD timings are 9 am to 1 pm.")).toBeInTheDocument();
  expect(screen.queryByText("CareBot is replying…")).not.toBeInTheDocument();
});

test("a failure shows an error and puts the message back in the box", async () => {
  api.post.mockRejectedValue({ response: { status: 429 } });
  const user = await openChat();

  await say(user, "book an appointment");

  expect(await screen.findByRole("alert")).toHaveTextContent("Too many attempts");
  expect(screen.getByLabelText("Message to CareBot")).toHaveValue("book an appointment");
});

test("the error clears on the next successful message", async () => {
  api.post.mockRejectedValueOnce({ response: { status: 500 } }).mockResolvedValue(reply());
  const user = await openChat();

  await say(user, "hello");
  await screen.findByRole("alert");
  await user.click(screen.getByRole("button", { name: "Send" }));

  await screen.findByText("OPD timings are 9 am to 1 pm.");
  expect(screen.queryByRole("alert")).not.toBeInTheDocument();
});

test("reply text is shown as text, never as markup", async () => {
  api.post.mockResolvedValue(reply({ reply: '<img src=x onerror="alert(1)"> hello' }));
  const user = await openChat();

  await say(user, "hi");

  expect(await screen.findByText('<img src=x onerror="alert(1)"> hello')).toBeInTheDocument();
  expect(document.querySelector("img")).toBeNull();
});

test.each([
  ["book_slot", 1],
  ["cancel_appointment", 1],
  ["reschedule_appointment", 1],
  ["find_slots", 0],
])("after %s the rest of the app is told %i time(s) to reload", async (tool, times) => {
  const listener = vi.fn();
  window.addEventListener(APPOINTMENTS_CHANGED, listener);
  api.post.mockResolvedValue(reply({ tools_called: [tool] }));
  const user = await openChat();

  await say(user, "yes");
  await screen.findByText("OPD timings are 9 am to 1 pm.");

  expect(listener).toHaveBeenCalledTimes(times);
  window.removeEventListener(APPOINTMENTS_CHANGED, listener);
});

test("the message box is limited to 500 characters", async () => {
  await openChat();

  expect(screen.getByLabelText("Message to CareBot")).toHaveAttribute("maxlength", "500");
});
