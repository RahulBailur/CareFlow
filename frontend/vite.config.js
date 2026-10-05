import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const backend = "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  build: { outDir: "build" },
  server: {
    // In development the API and Socket.IO live on the FastAPI server
    proxy: {
      "/api": backend,
      "/socket.io": { target: backend, ws: true },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.js",
  },
});
