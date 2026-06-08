"use client";

import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { AnalyticsResponse, Student } from "@/lib/types";
import { Badge, Card, EmptyState, ErrorBanner, Spinner } from "./ui";

export default function RegistryTab() {
  const [students, setStudents] = useState<Student[]>([]);
  const [sessions, setSessions] = useState<Record<string, number>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    try {
      const [studentList, analytics] = await Promise.all([
        api.students(),
        api.analytics(),
      ]);
      setStudents(studentList);
      const map: Record<string, number> = {};
      (analytics as AnalyticsResponse).leaderboard.forEach((s) => {
        map[s.student_id] = s.total_sessions;
      });
      setSessions(map);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Unknown error");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, [load]);

  if (loading) return <Spinner label="Loading registry…" />;
  if (error) return <ErrorBanner message={error} />;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-lg font-semibold text-slate-100">
          Enrolled students
        </h2>
        <p className="text-sm text-slate-400">
          {students.length} student{students.length === 1 ? "" : "s"} registered.
        </p>
      </div>

      <Card className="p-5">
        {students.length === 0 ? (
          <EmptyState message="No students enrolled yet. Use the Enroll button in the header." />
        ) : (
          <div className="overflow-hidden rounded-xl border border-white/5">
            <table className="w-full text-left text-sm">
              <thead className="bg-white/5 text-xs uppercase tracking-wider text-slate-400">
                <tr>
                  <th className="px-4 py-3">Student ID</th>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Enrolled at</th>
                  <th className="px-4 py-3">Sessions attended</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5">
                {students.map((s) => {
                  const attended = sessions[s.student_id] ?? 0;
                  return (
                    <tr key={s.student_id} className="hover:bg-white/[0.03]">
                      <td className="px-4 py-3 font-mono text-xs text-slate-400">
                        {s.student_id}
                      </td>
                      <td className="px-4 py-3 font-medium text-slate-100">
                        {s.name}
                      </td>
                      <td className="px-4 py-3 text-slate-300">
                        {s.enrolled_at ?? "—"}
                      </td>
                      <td className="px-4 py-3">
                        {attended === 0 ? (
                          <Badge tone="red">Never attended</Badge>
                        ) : (
                          <Badge tone="green">{attended}</Badge>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Card>
    </div>
  );
}
