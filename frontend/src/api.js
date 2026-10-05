import axios from "axios";

const TOKEN_KEY = "careflow_token";

// localStorage can be unavailable (private windows, blocked site data), so never let it throw
export const tokenStore = {
  get() {
    try {
      return localStorage.getItem(TOKEN_KEY);
    } catch {
      return null;
    }
  },
  set(token) {
    try {
      localStorage.setItem(TOKEN_KEY, token);
    } catch {
      /* the session simply won't survive a reload */
    }
  },
  clear() {
    try {
      localStorage.removeItem(TOKEN_KEY);
    } catch {
      /* nothing to clear */
    }
  },
};

const api = axios.create();

api.interceptors.request.use((config) => {
  const token = tokenStore.get();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

/** Turn an API failure into one sentence a person can act on. */
export function errorMessage(error, fallback = "Something went wrong. Please try again.") {
  const response = error?.response;
  if (!response) return "Could not reach the server. Check your connection and try again.";
  if (response.status === 429) return "Too many attempts. Please wait a minute and try again.";
  const detail = response.data?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg.replace(/^Value error, /, "");
  return fallback;
}

export default api;
