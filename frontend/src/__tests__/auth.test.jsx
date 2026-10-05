import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";

import api from "../api";
import DemoBanner from "../components/DemoBanner";
import ProtectedRoute from "../components/ProtectedRoute";
import { AuthProvider } from "../context/AuthContext";
import Login from "../pages/Login";
import Register from "../pages/Register";

vi.mock("../api", async (importOriginal) => ({
  ...(await importOriginal()),
  default: {
    get: vi.fn(),
    post: vi.fn(),
    interceptors: { response: { use: vi.fn(), eject: vi.fn() } },
  },
}));

const PATIENT = { id: "p1", name: "Asha Rao", email: "asha@example.com", role: "patient" };

function renderAt(path, routes) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AuthProvider>
        <Routes>{routes}</Routes>
      </AuthProvider>
    </MemoryRouter>,
  );
}

beforeEach(() => vi.clearAllMocks());

test("the demo banner warns against entering real health information", () => {
  render(<DemoBanner />);

  expect(screen.getByRole("note")).toHaveTextContent("do not enter real health information");
});

describe("Register", () => {
  const routes = (
    <>
      <Route path="/register" element={<Register />} />
      <Route path="/" element={<p>Home page</p>} />
    </>
  );

  async function fill(user, { phone = "9876543210", password = "long-enough-1" } = {}) {
    await user.type(screen.getByLabelText("Full name"), "Asha Rao");
    await user.type(screen.getByLabelText("Email"), "asha@example.com");
    await user.type(screen.getByLabelText("Mobile number"), phone);
    await user.type(screen.getByLabelText("Password"), password);
    await user.click(screen.getByRole("button", { name: "Create account" }));
  }

  test("offers no way to choose a role", () => {
    renderAt("/register", routes);

    expect(screen.queryByLabelText(/role/i)).not.toBeInTheDocument();
    expect(screen.queryByRole("combobox")).not.toBeInTheDocument();
    expect(screen.queryByText(/doctor|admin/i)).not.toBeInTheDocument();
  });

  test("rejects a phone number that is not a valid Indian mobile", async () => {
    renderAt("/register", routes);

    await fill(userEvent.setup(), { phone: "12345" });

    expect(screen.getByRole("alert")).toHaveTextContent("10-digit Indian mobile number");
    expect(api.post).not.toHaveBeenCalled();
  });

  test("rejects a short password", async () => {
    renderAt("/register", routes);

    await fill(userEvent.setup(), { password: "short" });

    expect(screen.getByRole("alert")).toHaveTextContent("at least 8 characters");
    expect(api.post).not.toHaveBeenCalled();
  });

  test("registers without sending a role, then goes home", async () => {
    api.post.mockResolvedValue({ data: { access_token: "token-1", user: PATIENT } });
    renderAt("/register", routes);

    await fill(userEvent.setup());

    expect(api.post).toHaveBeenCalledWith("/api/auth/register", {
      name: "Asha Rao",
      email: "asha@example.com",
      phone: "9876543210",
      password: "long-enough-1",
    });
    expect(await screen.findByText("Home page")).toBeInTheDocument();
    expect(localStorage.getItem("careflow_token")).toBe("token-1");
  });

  test("shows the server's message when the email is already registered", async () => {
    api.post.mockRejectedValue({
      response: { status: 409, data: { detail: "Email already registered" } },
    });
    renderAt("/register", routes);

    await fill(userEvent.setup());

    expect(await screen.findByRole("alert")).toHaveTextContent("Email already registered");
  });
});

describe("Login", () => {
  const routes = (
    <>
      <Route path="/login" element={<Login />} />
      <Route path="/" element={<p>Home page</p>} />
    </>
  );

  async function submit(user) {
    await user.type(screen.getByLabelText("Email"), "asha@example.com");
    await user.type(screen.getByLabelText("Password"), "a-password");
    await user.click(screen.getByRole("button", { name: "Log in" }));
  }

  test("logs in and goes home", async () => {
    api.post.mockResolvedValue({ data: { access_token: "token-2", user: PATIENT } });
    renderAt("/login", routes);

    await submit(userEvent.setup());

    expect(api.post).toHaveBeenCalledWith("/api/auth/login", {
      email: "asha@example.com",
      password: "a-password",
    });
    expect(await screen.findByText("Home page")).toBeInTheDocument();
  });

  test("shows why a login failed and lets the user try again", async () => {
    api.post.mockRejectedValue({
      response: { status: 401, data: { detail: "Invalid email or password" } },
    });
    renderAt("/login", routes);

    await submit(userEvent.setup());

    expect(await screen.findByRole("alert")).toHaveTextContent("Invalid email or password");
    expect(screen.getByRole("button", { name: "Log in" })).toBeEnabled();
  });

  test("explains rate limiting in plain words", async () => {
    api.post.mockRejectedValue({ response: { status: 429, data: { error: "Rate limit" } } });
    renderAt("/login", routes);

    await submit(userEvent.setup());

    expect(await screen.findByRole("alert")).toHaveTextContent("Too many attempts");
  });
});

describe("ProtectedRoute", () => {
  const routes = (
    <>
      <Route path="/login" element={<p>Login page</p>} />
      <Route path="/" element={<p>Home page</p>} />
      <Route
        path="/book"
        element={
          <ProtectedRoute roles={["patient"]}>
            <p>Booking page</p>
          </ProtectedRoute>
        }
      />
    </>
  );

  test("sends a signed-out visitor to the login page", () => {
    renderAt("/book", routes);

    expect(screen.getByText("Login page")).toBeInTheDocument();
  });

  test("restores a stored session and shows the page", async () => {
    localStorage.setItem("careflow_token", "stored-token");
    api.get.mockResolvedValue({ data: PATIENT });

    renderAt("/book", routes);

    expect(await screen.findByText("Booking page")).toBeInTheDocument();
    expect(api.get).toHaveBeenCalledWith("/api/auth/me");
  });

  test("keeps a doctor out of patient-only pages", async () => {
    localStorage.setItem("careflow_token", "stored-token");
    api.get.mockResolvedValue({ data: { ...PATIENT, role: "doctor" } });

    renderAt("/book", routes);

    expect(await screen.findByText("Home page")).toBeInTheDocument();
    expect(screen.queryByText("Booking page")).not.toBeInTheDocument();
  });

  test("signs out when the stored token is no longer valid", async () => {
    localStorage.setItem("careflow_token", "expired-token");
    api.get.mockRejectedValue({ response: { status: 401 } });

    renderAt("/book", routes);

    expect(await screen.findByText("Login page")).toBeInTheDocument();
    expect(localStorage.getItem("careflow_token")).toBeNull();
  });
});
