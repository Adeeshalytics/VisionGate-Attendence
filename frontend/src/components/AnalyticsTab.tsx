"use client";

import { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import type { AnalyticsResponse } from "@/lib/types";
import { Card, EmptyState, ErrorBanner, Spinner } from "./ui";

const tooltipStyle = {
  background: "#0f1530",
  border: "1px solid rgba(255,255,255,0.1)",
  borderRadius: 12,
  color: "#e8ecf5",
};

export default function AnalyticsTab() {
  const [data, setData] = useState<AnalyticsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const res = await api.analytics(14);
      setData(res);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  if (loading) return <Spinner label="Loading analytics…" />;
  if (error) return <ErrorBanner message={error} />;
  if (!data) return null;

  const leaderboard = data.leaderboard.slice(0, 10);

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-slate-100">Analytics</h2>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="p-5">
          <h3 className="mb-4 text-sm font-semibold text-slate-300">
            Top students by attendance
          </h3>
          {leaderboard.length === 0 ? (
            <EmptyState message="No attendance data yet." />
          ) : (
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={leaderboard}
                  layout="vertical"
                  margin={{ left: 20, right: 20 }}
                >
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="rgba(255,255,255,0.06)"
                  />
                  <XAxis
                    type="number"
                    allowDecimals={false}
                    stroke="#64748b"
                    fontSize={12}
                  />
                  <YAxis
                    type="category"
                    dataKey="name"
                    width={110}
                    stroke="#94a3b8"
                    fontSize={12}
                  />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
                  <Bar
                    dataKey="total_sessions"
                    fill="#38bdf8"
                    radius={[0, 6, 6, 0]}
                    name="Sessions"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </Card>

        <Card className="p-5">
          <h3 className="mb-4 text-sm font-semibold text-slate-300">
            Daily attendance — last 14 days
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={data.daily_counts}
                margin={{ left: 0, right: 20 }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="rgba(255,255,255,0.06)"
                />
                <XAxis
                  dataKey="date"
                  stroke="#64748b"
                  fontSize={11}
                  tickFormatter={(d: string) => d.slice(5)}
                />
                <YAxis allowDecimals={false} stroke="#64748b" fontSize={12} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line
                  type="monotone"
                  dataKey="count"
                  stroke="#a78bfa"
                  strokeWidth={2}
                  dot={{ r: 3, fill: "#a78bfa" }}
                  name="Records"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>

      <Card className="p-5">
        <h3 className="mb-1 text-sm font-semibold text-slate-300">
          Students below 80% attendance
        </h3>
        <p className="mb-4 text-xs text-slate-500">
          Based on {data.total_sessions_held} distinct session day
          {data.total_sessions_held === 1 ? "" : "s"} held.
        </p>
        {data.low_attendance.length === 0 ? (
          <EmptyState message="No students below the threshold (or not enough data yet)." />
        ) : (
          <div className="overflow-hidden rounded-xl border border-white/5">
            <table className="w-full text-left text-sm">
              <thead className="bg-white/5 text-xs uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-4 py-3">Student ID</th>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Attended</th>
                  <th className="px-4 py-3">Held</th>
                  <th className="px-4 py-3">Rate</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {data.low_attendance.map((r) => (
                  <tr key={r.student_id} className="hover:bg-white/[0.03]">
                    <td className="px-4 py-3 font-mono text-xs text-slate-400">
                      {r.student_id}
                    </td>
                    <td className="px-4 py-3 font-medium text-slate-100">
                      {r.name}
                    </td>
                    <td className="px-4 py-3 text-slate-300">
                      {r.sessions_attended}
                    </td>
                    <td className="px-4 py-3 text-slate-300">
                      {r.sessions_held}
                    </td>
                    <td className="px-4 py-3 font-semibold text-rose-300">
                      {r.attendance_rate.toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
