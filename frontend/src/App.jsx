import { Navigate, Route, Routes } from "react-router-dom";

import DemoBanner from "./components/DemoBanner";
import Navbar from "./components/Navbar";
import ProtectedRoute from "./components/ProtectedRoute";
import { useAuth } from "./context/AuthContext";
import AdminHome from "./pages/AdminHome";
import BookAppointment from "./pages/BookAppointment";
import DoctorDashboard from "./pages/DoctorDashboard";
import Login from "./pages/Login";
import PatientDashboard from "./pages/PatientDashboard";
import Register from "./pages/Register";
import VisitHistory from "./pages/VisitHistory";

const HOME = { patient: PatientDashboard, doctor: DoctorDashboard, admin: AdminHome };

function Home() {
  const { user } = useAuth();
  const Page = HOME[user.role];
  return <Page />;
}

export default function App() {
  return (
    <>
      <DemoBanner />
      <Navbar />
      <main>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Home />
              </ProtectedRoute>
            }
          />
          <Route
            path="/book"
            element={
              <ProtectedRoute roles={["patient"]}>
                <BookAppointment />
              </ProtectedRoute>
            }
          />
          <Route
            path="/history"
            element={
              <ProtectedRoute roles={["patient"]}>
                <VisitHistory />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </main>
    </>
  );
}
