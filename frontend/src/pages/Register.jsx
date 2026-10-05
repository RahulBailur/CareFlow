import { useState } from "react";
import { Link, Navigate } from "react-router-dom";

import { errorMessage } from "../api";
import { useAuth } from "../context/AuthContext";

const PHONE = /^[6-9]\d{9}$/;

function validate({ name, phone, password }) {
  if (name.trim().length < 2) return "Enter your full name.";
  if (!PHONE.test(phone))
    return "Enter a 10-digit Indian mobile number starting with 6, 7, 8 or 9.";
  if (password.length < 8) return "Password must be at least 8 characters.";
  return "";
}

/** Patient registration. There is deliberately no role choice: staff accounts are seeded. */
export default function Register() {
  const { user, register } = useAuth();
  const [form, setForm] = useState({ name: "", email: "", phone: "", password: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  if (user) return <Navigate to="/" replace />;

  const update = (field) => (event) => setForm({ ...form, [field]: event.target.value });

  async function handleSubmit(event) {
    event.preventDefault();
    const problem = validate(form);
    setError(problem);
    if (problem) return;
    setBusy(true);
    try {
      await register({ ...form, name: form.name.trim(), email: form.email.trim() });
    } catch (err) {
      setError(errorMessage(err, "Could not create the account."));
      setBusy(false);
    }
  }

  return (
    <div className="auth-page">
      <form className="card form" onSubmit={handleSubmit} noValidate>
        <h1>Create a patient account</h1>
        <p className="muted">Use made-up details. This is a demo, not a real hospital.</p>

        <label htmlFor="name">Full name</label>
        <input id="name" autoComplete="name" value={form.name} onChange={update("name")} />

        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          autoComplete="email"
          value={form.email}
          onChange={update("email")}
        />

        <label htmlFor="phone">Mobile number</label>
        <input
          id="phone"
          type="tel"
          inputMode="numeric"
          autoComplete="tel-national"
          maxLength={10}
          placeholder="10 digits"
          value={form.phone}
          onChange={update("phone")}
        />

        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          autoComplete="new-password"
          aria-describedby="password-hint"
          value={form.password}
          onChange={update("password")}
        />
        <p id="password-hint" className="hint">
          At least 8 characters.
        </p>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <button type="submit" className="button primary" disabled={busy}>
          {busy ? "Creating account…" : "Create account"}
        </button>
        <p className="muted">
          Already registered? <Link to="/login">Log in</Link>
        </p>
      </form>
    </div>
  );
}
