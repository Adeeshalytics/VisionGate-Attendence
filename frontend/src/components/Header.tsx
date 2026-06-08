"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import EnrollModal from "./EnrollModal";
import EnrollProgress from "./EnrollProgress";

type Status = "checking" | "online" | "offline";

export default function Header() {
  const [status, setStatus] = useState<Status>("checking");
  const [toast, setToast] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [enrolling, setEnrolling] = useState<{ name: string; studentId: string } | null>(
    null,
  );
  const [windowRunning, setWindowRunning] = useState(false);

  useEffect(() => {
    let active = true;
    const check = async () => {
      try {
        await api.health();
        if (active) setStatus("online");
        try {
          const rs = await api.recognizeStatus();
          if (active) setWindowRunning(rs.running);
        } catch {
          /* ignore */
        }
      } catch {
        if (active) setStatus("offline");
      }
    };
    check();
    const id = setInterval(check, 5000);
    return () => {
      active = false;
      clearInterval(id);
    };
  }, []);

  const showToast = (msg: string) => {
    setToast(msg);
    setTimeout(() => setToast(null), 4000);
  };

  const toggleWindow = async () => {
    setBusy("recognize");
    try {
      if (windowRunning) {
        const res = await api.stopRecognize();
        setWindowRunning(false);
        showToast(res.message);
      } else {
        const res = await api.launchRecognize();
        if (res.started) setWindowRunning(true);
        showToast(res.message);
      }
    } catch (e) {
      showToast(e instanceof Error ? e.message : "Action failed");
    } finally {
      setBusy(null);
    }
  };

  const dot =
    status === "online"
      ? "bg-emerald-400"
      : status === "offline"
        ? "bg-rose-500"
        : "bg-amber-400";

  return (
    <header className="sticky top-0 z-20 border-b border-white/10 bg-[#0b1020]/80 backdrop-blur">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-sky-500 to-violet-500 text-lg font-bold text-white shadow-lg">
            VG
          </div>
          <div>
            <h1 className="text-base font-semibold leading-tight text-slate-100">
              VisionGate
            </h1>
            <p className="text-xs text-slate-400">Attendance Dashboard</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <span className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300">
            <span className={`h-2 w-2 rounded-full ${dot}`} />
            {status}
          </span>

          <button
            onClick={() => setEnrollOpen(true)}
            disabled={status !== "online"}
            className="rounded-lg border border-white/10 bg-white/5 px-4 py-2 text-sm font-medium text-slate-100 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Enroll student
          </button>
          <button
            onClick={toggleWindow}
            disabled={status !== "online" || busy !== null}
            className={`rounded-lg px-4 py-2 text-sm font-semibold text-white shadow-lg transition hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40 ${
              windowRunning
                ? "bg-gradient-to-r from-rose-500 to-rose-600"
                : "bg-gradient-to-r from-sky-500 to-violet-500"
            }`}
          >
            {busy === "recognize"
              ? windowRunning
                ? "Stopping…"
                : "Launching…"
              : windowRunning
                ? "Stop window"
                : "Session window"}
          </button>
        </div>
      </div>

      {toast ? (
        <div className="mx-auto max-w-7xl px-6 pb-3">
          <div className="rounded-lg border border-sky-400/30 bg-sky-500/10 px-4 py-2 text-sm text-sky-200">
            {toast}
          </div>
        </div>
      ) : null}

      {enrolling ? (
        <EnrollProgress
          name={enrolling.name}
          studentId={enrolling.studentId}
          onDismiss={() => setEnrolling(null)}
        />
      ) : null}

      <EnrollModal
        open={enrollOpen}
        onClose={() => setEnrollOpen(false)}
        onLaunched={(name, studentId) => setEnrolling({ name, studentId })}
      />
    </header>
  );
}
