// The hospital runs on Indian Standard Time, so every time is shown in IST wherever the viewer is.
const TIME_ZONE = "Asia/Kolkata";

const timeFormat = new Intl.DateTimeFormat("en-IN", {
  timeZone: TIME_ZONE,
  hour: "numeric",
  minute: "2-digit",
  hour12: true,
});

const dateFormat = new Intl.DateTimeFormat("en-IN", {
  timeZone: TIME_ZONE,
  weekday: "short",
  day: "numeric",
  month: "short",
  year: "numeric",
});

// en-CA formats as YYYY-MM-DD, which is what <input type="date"> and the API expect
const isoDayFormat = new Intl.DateTimeFormat("en-CA", { timeZone: TIME_ZONE });

export const formatTime = (value) => timeFormat.format(new Date(value));
export const formatDate = (value) => dateFormat.format(new Date(value));
export const isoDay = (value) => isoDayFormat.format(new Date(value));
export const todayIso = () => isoDay(Date.now());
export const isToday = (value) => isoDay(value) === todayIso();

export function addDaysIso(isoDate, days) {
  const date = new Date(`${isoDate}T00:00:00Z`);
  date.setUTCDate(date.getUTCDate() + days);
  return date.toISOString().slice(0, 10);
}
