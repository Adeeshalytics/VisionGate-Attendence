"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AttendanceRecord } from "@/lib/types";
import ConfirmDialog from "./ConfirmDialog";
import {
  Badge,
  Card,
  EmptyState,
  ErrorBanner,
  Spinner,
  confidenceTone,
} from "./ui";

function todayISO(): string {
  return new Date().toISOString().slice(0, 10);
}

export default function HistoryTab() {
  const [day, setDay] = useState<string>(todayISO());
  const [rows, setRows] = useState<AttendanceRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [clearing, setClearing] = useState(false);

  const load = useCallback(async (d: string) => {
    setLoading(true);
    try {
      const res = await api.attendanceByDate(d);
      setRows(res);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(day);
  }, [day, load]);

  const handleClear = async () => {
    setClearing(true);
    try {
      await api.clearAttendance(day);
      setConfirmOpen(false);
      await load(day);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to clear records.");
    } finally {
      setClearing(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">
            Attendance history
          </h2>
          <p className="text-sm text-slate-400">Pick a date to view records.</p>
        </div>
        <div className="flex items-end gap-3">
          <label className="flex flex-col gap-1 text-xs text-slate-400">
            Date
            <input
              type="date"
              value={day}
              max={todayISO()}
              onChange={(e) => setDay(e.target.value)}
              className="rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 outline-none focus:border-sky-400/60"
            />
          </label>
          <a
            href={api.exportUrl(day)}
            className="rounded-lg bg-sky-500/90 px-4 py-2 text-sm font-medium text-white transition hover:bg-sky-400"
          >
            Download CSV
          </a>
          <button
            onClick={() => setConfirmOpen(true)}
            disabled={rows.length === 0}
            className="rounded-lg border border-rose-400/30 bg-rose-500/15 px-4 py-2 text-sm font-medium text-rose-100 transition hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Clear records
          </button>
        </div>
      </div>

      <Card className="p-5">
        {loading ? (
          <Spinner label={`Loading ${day}…`} />
        ) : error ? (
          <ErrorBanner message={error} />
        ) : rows.length === 0 ? (
          <EmptyState message={`No attendance recorded on ${day}.`} />
        ) : (
          <div className="overflow-hidden rounded-xl border border-white/5">
            <table className="w-full text-left text-sm">
              <thead className="bg-white/5 text-xs uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-4 py-3">Student ID</th>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Time</th>
                  <th className="px-4 py-3">Confidence</th>
                  <th className="px-4 py-3">Session</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {rows.map((r) => (
                  <tr key={`${r.student_id}-${r.id}`} className="hover:bg-white/[0.03]">
                    <td className="px-4 py-3 font-mono text-xs text-slate-400">
                      {r.student_id}
                    </td>
                    <td className="px-4 py-3 font-medium text-slate-100">
                      {r.name}
                    </td>
                    <td className="px-4 py-3 text-slate-300">{r.time}</td>
                    <td className="px-4 py-3">
                      <Badge tone={confidenceTone(r.confidence)}>
                        {r.confidence !== null
                          ? `${(r.confidence * 100).toFixed(1)}%`
                          : "—"}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-slate-400">
                      {r.session ?? "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>

      <ConfirmDialog
        open={confirmOpen}
        title="Clear attendance records"
        message={`This will permanently delete all ${rows.length} attendance record(s) for ${day}. Enrolled students are not affected.`}
        confirmLabel="Clear records"
        busy={clearing}
        onConfirm={handleClear}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
}
