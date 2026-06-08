"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Cell,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { api } from "@/lib/api";
import type { TodayOverview } from "@/lib/types";
import ConfirmDialog from "./ConfirmDialog";
import {
  Badge,
  Card,
  EmptyState,
  ErrorBanner,
  Spinner,
  StatCard,
  confidenceTone,
} from "./ui";

const REFRESH_MS = 5000;

export default function TodayTab() {
  const [data, setData] = useState<TodayOverview | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [updatedAt, setUpdatedAt] = useState<string>("");
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [clearing, setClearing] = useState(false);

  const load = useCallback(async () => {
    try {
      const res = await api.today();
      setData(res);
      setError(null);
      setUpdatedAt(new Date().toLocaleTimeString());
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, REFRESH_MS);
    return () => clearInterval(id);
  }, [load]);

  const handleClear = async () => {
    if (!data) return;
    setClearing(true);
    try {
      await api.clearAttendance(data.date);
      setConfirmOpen(false);
      await load();
    } catch {
      // surfaced via the banner on next load if persistent
    } finally {
      setClearing(false);
    }
  };

  if (loading) return <Spinner label="Loading today's attendance…" />;
  if (error) return <ErrorBanner message={error} />;
  if (!data) return null;

  const absent = Math.max(0, data.total_enrolled - data.present_count);
  const pieData = [
    { name: "Present", value: data.present_count },
    { name: "Absent", value: absent },
  ];
  const COLORS = ["#34d399", "#f43f5e"];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-slate-100">
          Today — {data.date}
        </h2>
        <span className="text-xs text-slate-500">
          Live · updated {updatedAt}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        <StatCard
          label="Present today"
          value={data.present_count}
          accent="text-emerald-300"
        />
        <StatCard
          label="Total enrolled"
          value={data.total_enrolled}
          accent="text-sky-300"
        />
        <StatCard
          label="Attendance rate"
          value={`${data.attendance_rate.toFixed(1)}%`}
          accent="text-violet-300"
        />
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="p-5 lg:col-span-2">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-300">
              Present students
            </h3>
            <button
              onClick={() => setConfirmOpen(true)}
              disabled={data.records.length === 0}
              className="rounded-lg border border-rose-400/30 bg-rose-500/15 px-3 py-1.5 text-xs font-medium text-rose-100 transition hover:bg-rose-500/25 disabled:cursor-not-allowed disabled:opacity-40"
            >
              Clear today
            </button>
          </div>
          {data.records.length === 0 ? (
            <EmptyState message="No attendance recorded yet today." />
          ) : (
            <div className="overflow-hidden rounded-xl border border-white/5">
              <table className="w-full text-left text-sm">
                <thead className="bg-white/5 text-xs uppercase tracking-wider text-slate-400">
                  <tr>
                    <th className="px-4 py-3">Name</th>
                    <th className="px-4 py-3">Student ID</th>
                    <th className="px-4 py-3">Time</th>
                    <th className="px-4 py-3">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {data.records.map((r) => (
                    <tr key={`${r.student_id}-${r.id}`} className="hover:bg-white/[0.03]">
                      <td className="px-4 py-3 font-medium text-slate-100">
                        {r.name}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs text-slate-400">
                        {r.student_id}
                      </td>
                      <td className="px-4 py-3 text-slate-300">{r.time}</td>
                      <td className="px-4 py-3">
                        <Badge tone={confidenceTone(r.confidence)}>
                          {r.confidence !== null
                            ? `${(r.confidence * 100).toFixed(1)}%`
                            : "—"}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>

        <Card className="p-5">
          <h3 className="mb-2 text-sm font-semibold text-slate-300">
            Present vs Absent
          </h3>
          {data.total_enrolled === 0 ? (
            <EmptyState message="Enroll students to see the breakdown." />
          ) : (
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={55}
                    outerRadius={85}
                    paddingAngle={3}
                  >
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={COLORS[i]} stroke="transparent" />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      background: "#0f1530",
                      border: "1px solid rgba(255,255,255,0.1)",
                      borderRadius: 12,
                      color: "#e8ecf5",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="-mt-4 flex justify-center gap-6 text-xs">
                <span className="flex items-center gap-2 text-slate-300">
                  <span className="h-3 w-3 rounded-full bg-emerald-400" /> Present (
                  {data.present_count})
                </span>
                <span className="flex items-center gap-2 text-slate-300">
                  <span className="h-3 w-3 rounded-full bg-rose-500" /> Absent (
                  {absent})
                </span>
              </div>
            </div>
          )}
        </Card>
      </div>

      <ConfirmDialog
        open={confirmOpen}
        title="Clear today's attendance"
        message={`This will permanently delete all ${data.records.length} attendance record(s) for ${data.date}. Enrolled students are not affected.`}
        confirmLabel="Clear today"
        busy={clearing}
        onConfirm={handleClear}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  );
}
