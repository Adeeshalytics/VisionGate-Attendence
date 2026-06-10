/**
 * Shared API types for the VisionGate frontend.
 *
 * These mirror the Pydantic response models defined in `api/main.py`.
 * Keep them in sync with that backend contract.
 */

export interface Student {
  student_id: string;
  name: string;
  enrolled_at: string | null;
}

export interface AttendanceRecord {
  id?: number | null;
  student_id: string;
  name: string;
  date: string;
  time: string;
  confidence: number | null;
  session?: string | null;
}

export interface TodayOverview {
  date: string;
  present_count: number;
  total_enrolled: number;
  /** Already a percentage (0–100), rounded to 1 dp by the backend. */
  attendance_rate: number;
  records: AttendanceRecord[];
}

export interface StudentSummary {
  student_id: string;
  name: string;
  total_sessions: number;
  enrolled_at?: string | null;
}

export interface DailyCount {
  date: string;
  count: number;
}

export interface LowAttendance {
  student_id: string;
  name: string;
  sessions_attended: number;
  sessions_held: number;
  /** Already a percentage (0–100). */
  attendance_rate: number;
}

export interface AnalyticsResponse {
  total_students: number;
  total_sessions_held: number;
  leaderboard: StudentSummary[];
  daily_counts: DailyCount[];
  low_attendance: LowAttendance[];
}

export interface RecognizedStudent {
  student_id: string;
  name: string;
  time: string;
  confidence: number | null;
}

export interface StreamStatus {
  running: boolean;
  session: string;
  error: string | null;
  recognized_count: number;
  students: RecognizedStudent[];
}

export interface LaunchResponse {
  started: boolean;
  message: string;
  pid?: number | null;
}

export interface RecognizeStatus {
  running: boolean;
  pid?: number | null;
}

export interface StreamStartResponse {
  started: boolean;
  message: string;
  session?: string;
}

export interface SimpleMessage {
  message: string;
  [key: string]: unknown;
}
