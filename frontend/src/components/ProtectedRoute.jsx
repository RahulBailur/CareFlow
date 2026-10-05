import { Navigate } from "react-router-dom";

import { useAuth } from "../context/AuthContext";

/** Render children only for a signed-in user, optionally limited to certain roles. */
export default function ProtectedRoute({ roles, children }) {
  const { user, loading } = useAuth();

  if (loading) return <p className="muted">Loading…</p>;
  if (!user) return <Navigate to="/login" replace />;
  if (roles && !roles.includes(user.role)) return <Navigate to="/" replace />;
  return children;
}
