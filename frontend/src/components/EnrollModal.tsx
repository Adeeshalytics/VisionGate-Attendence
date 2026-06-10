"use client";

import { useState } from "react";
import { createPortal } from "react-dom";
import { api } from "@/lib/api";

export default function EnrollModal({
  open,
  onClose,
  onLaunched,
}: {
  open: boolean;
  onClose: () => void;
  onLaunched: (name: string, studentId: string) => void;
}) {
  const [name, setName] = useState("");
  const [studentId, setStudentId] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const reset = () => {
    setName("");
    setStudentId("");
    setError(null);
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name.trim() || !studentId.trim()) {
      setError("Both name and student ID are required.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const cleanName = name.trim();
      const cleanId = studentId.trim();
      await api.launchEnroll(cleanName, cleanId);
      onLaunched(cleanName, cleanId);
      reset();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start enrollment.");
    } finally {
      setSubmitting(false);
    }
  };

  return createPortal(
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-white/10 bg-[#0f1530] p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold text-slate-100">Enroll new student</h2>
        <p className="mt-1 text-sm text-slate-400">
          Enter the details, then look at the camera. A webcam window will open
          to capture the face.
        </p>

        <form onSubmit={submit} className="mt-5 space-y-4">
          <label className="block">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Full name
            </span>
            <input
              autoFocus
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Ada Lovelace"
              className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-100 outline-none focus:border-sky-400/60"
            />
          </label>

          <label className="block">
            <span className="text-xs font-medium uppercase tracking-wider text-slate-400">
              Student ID
            </span>
            <input
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              placeholder="e.g. EG2021001"
              className="mt-1 w-full rounded-lg border border-white/10 bg-white/5 px-3 py-2 font-mono text-sm text-slate-100 outline-none focus:border-sky-400/60"
            />
          </label>

          {error ? (
            <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              {error}
            </div>
          ) : null}

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={submitting}
              className="rounded-lg bg-gradient-to-r from-sky-500 to-violet-500 px-4 py-2 text-sm font-semibold text-white shadow-lg transition hover:opacity-90 disabled:opacity-50"
            >
              {submitting ? "Starting…" : "Start enrollment"}
            </button>
          </div>
        </form>
      </div>
    </div>,
    document.body
  );
}
