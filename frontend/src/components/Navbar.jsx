import { Link, NavLink } from "react-router-dom";

import { useAuth } from "../context/AuthContext";

const LINKS = {
  patient: [
    { to: "/", label: "My appointments" },
    { to: "/book", label: "Book" },
    { to: "/history", label: "Visit history" },
  ],
  doctor: [{ to: "/", label: "Today's queue" }],
  admin: [{ to: "/", label: "Overview" }],
};

export default function Navbar() {
  const { user, logout } = useAuth();

  return (
    <header className="navbar">
      <Link to="/" className="brand">
        CareFlow
      </Link>
      {user ? (
        <>
          <nav aria-label="Main">
            {LINKS[user.role].map((link) => (
              <NavLink key={link.to} to={link.to} end>
                {link.label}
              </NavLink>
            ))}
          </nav>
          <div className="navbar-user">
            <span>
              {user.name} <span className="role-tag">{user.role}</span>
            </span>
            <button type="button" className="button quiet" onClick={logout}>
              Log out
            </button>
          </div>
        </>
      ) : (
        <nav aria-label="Main">
          <NavLink to="/login">Log in</NavLink>
          <NavLink to="/register">Register</NavLink>
        </nav>
      )}
    </header>
  );
}
