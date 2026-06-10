/**
 * Typed client for the VisionGate FastAPI backend (`api/main.py`).
 *
 * The base URL defaults to http://localhost:8000 and can be overridden with
 * the NEXT_PUBLIC_API_URL environment variable (e.g. in `.env.local`).
 */

import type {
  AnalyticsResponse,
  AttendanceRecord,
  LaunchResponse,
  RecognizeStatus,
  Student,
  StreamStartResponse,
  StreamStatus,
  TodayOverview,
} from "./types";

const BASE =
  (process.env.NEXT_PUBLIC_API_URL?.replace(/\/+$/, "") as string | undefined) ||
  "http://localhost:8000";

/** Perform a JSON request, throwing an Error with the backend detail on failure. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new Error("Cannot reach the API server. Is it running on :8000?");
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status})`;
    try {
      const body = await res.json();
      if (body && typeof body.detail === "string") detail = body.detail;
    } catch {
      /* response had no JSON body */
    }
    throw new Error(detail);
  }

  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

export const api = {
  /** Base URL the client is talking to (useful for debugging). */
  baseUrl: BASE,

  // -- meta ----------------------------------------------------------------
  health: () => request<{ status: string; db_path: string }>("/health"),

  // -- students ------------------------------------------------------------
  students: () => request<Student[]>("/students"),
  deleteAllStudents: () =>
    request<Record<string, unknown>>("/students", { method: "DELETE" }),

  // -- attendance ----------------------------------------------------------
  today: () => request<TodayOverview>("/attendance/today"),
  attendanceByDate: (day: string) =>
    request<AttendanceRecord[]>(`/attendance/${day}`),
  clearAttendance: (day: string) =>
    request<{ date: string; removed: number }>(`/attendance/${day}`, {
      method: "DELETE",
    }),

  // -- analytics -----------------------------------------------------------
  analytics: (days = 14) =>
    request<AnalyticsResponse>(`/analytics?days=${days}`),

  // -- CSV export (used directly as a download href) -----------------------
  exportUrl: (day: string) => `${BASE}/export/${day}`,

  // -- native window launch (enrollment / recognition) ---------------------
  launchEnroll: (name: string, student_id: string) =>
    request<LaunchResponse>("/launch/enroll", {
      method: "POST",
      body: JSON.stringify({ name, student_id }),
    }),
  launchRecognize: () =>
    request<LaunchResponse>("/launch/recognize", { method: "POST" }),
  stopRecognize: () =>
    request<LaunchResponse>("/launch/recognize/stop", { method: "POST" }),
  recognizeStatus: () =>
    request<RecognizeStatus>("/launch/recognize/status"),

  // -- in-dashboard MJPEG recognition stream -------------------------------
  streamStart: () =>
    request<StreamStartResponse>("/stream/start", { method: "POST" }),
  streamStop: () =>
    request<{ stopped: boolean; message: string }>("/stream/stop", {
      method: "POST",
    }),
  streamStatus: () => request<StreamStatus>("/stream/status"),
  /** MJPEG endpoint, used as an <img> src. */
  streamVideoUrl: () => `${BASE}/stream/video`,
};
