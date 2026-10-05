import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { io } from "socket.io-client";

import { useAuth } from "./AuthContext";

const QUEUE_EVENT = "queue:update";

const QueueSocketContext = createContext(null);

/** One authenticated Socket.IO connection per signed-in session, shared by every screen. */
export function QueueSocketProvider({ children }) {
  const { token } = useAuth();
  const listeners = useRef(new Set());
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    if (!token) return;
    const socket = io({ auth: { token }, transports: ["websocket"] });
    socket.on("connect", () => setConnected(true));
    socket.on("disconnect", () => setConnected(false));
    socket.on(QUEUE_EVENT, (update) => listeners.current.forEach((listener) => listener(update)));
    return () => {
      socket.disconnect();
      setConnected(false);
    };
  }, [token]);

  const value = useMemo(
    () => ({
      connected,
      subscribe(listener) {
        listeners.current.add(listener);
        return () => listeners.current.delete(listener);
      },
    }),
    [connected],
  );
  return <QueueSocketContext.Provider value={value}>{children}</QueueSocketContext.Provider>;
}

/** Call `handler` with every live queue update the server sends this user. */
export function useQueueUpdates(handler) {
  const context = useContext(QueueSocketContext);
  const latest = useRef(handler);
  useEffect(() => {
    latest.current = handler;
  });
  const subscribe = context?.subscribe;
  useEffect(() => subscribe?.((update) => latest.current(update)), [subscribe]);
  return { connected: context?.connected ?? false };
}
