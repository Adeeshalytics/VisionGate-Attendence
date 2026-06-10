"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Card, ErrorBanner } from "./ui";

interface RecognizedStudent {
  student_id: string;
  name: string;
  time: string;
  confidence: number | null;
}

interface StreamState {
  running: boolean;
  session: string;
  error: string | null;
  recognized_count: number;
  students: RecognizedStudent[];
}

export default function LiveSessionTab() {
  const [state, setState] = useState<StreamState | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [streamSrc, setStreamSrc] = useState<string | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const s = await api.streamStatus();
      setState(s);
      // If a session is running, make sure the video element is pointed at
      // the stream (cache-busted so it reconnects after a restart).
      if (s.running) {
        setStreamSrc((prev) => prev ?? `${api.streamVideoUrl()}?t=${Date.now()}`);
      } else {
        setStreamSrc(null);
      }
    } catch {
      // ignore transient polling errors
    }
  }, []);

  useEffect(() => {
    refreshStatus();
    const id = setInterval(refreshStatus, 3000);
    return () => clearInterval(id);
  }, [refreshStatus]);

  const start = async () => {
    setBusy(true);
    setActionError(null);
    try {
      const res = await api.streamStart();
      if (!res.started) {
        setActionError(res.message);
      } else {
        setStreamSrc(`${api.streamVideoUrl()}?t=${Date.now()}`);
      }
      await refreshStatus();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "Failed to start session.");
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    setActionError(null);
    try {
      await api.streamStop();
      setStreamSrc(null);
      await refreshStatus();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "Failed to stop session.");
    } finally {
      setBusy(false);
    }
  };

  const running = state?.running ?? false;

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-slate-100">Live session</h2>
          <p className="text-sm text-slate-400">
            Runs recognition in the dashboard. Recognized students are marked
            present automatically.
          </p>
        </div>
        <div className="flex items-center gap-3">
          {running ? (
            <span className="flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1.5 text-xs text-emerald-200">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
              Live · {state?.recognized_count ?? 0} recognized
            </span>
          ) : (
            <span className="flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-300">
              <span className="h-2 w-2 rounded-full bg-slate-500" />
              Stopped
            </span>
          )}
          {running ? (
            <button
              onClick={stop}
              disabled={busy}
              className="rounded-lg border border-rose-400/30 bg-rose-500/15 px-4 py-2 text-sm font-semibold text-rose-100 transition hover:bg-rose-500/25 disabled:opacity-50"
            >
              {busy ? "Stopping…" : "Stop session"}
            </button>
          ) : (
            <button
              onClick={start}
              disabled={busy}
              className="rounded-lg bg-gradient-to-r from-sky-500 to-violet-500 px-4 py-2 text-sm font-semibold text-white shadow-lg transition hover:opacity-90 disabled:opacity-50"
            >
              {busy ? "Starting…" : "Start session"}
            </button>
          )}
        </div>
      </div>

      {actionError ? <ErrorBanner message={actionError} /> : null}
      {state?.error && !running ? <ErrorBanner message={state.error} /> : null}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <Card className="overflow-hidden p-0 lg:col-span-2">
          <div className="relative flex aspect-video w-full items-center justify-center bg-black">
            {streamSrc ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img
                ref={imgRef}
                src={streamSrc}
                alt="Live recognition feed"
                className="h-full w-full object-contain"
              />
            ) : (
              <div className="flex flex-col items-center gap-3 text-slate-500">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  fill="none"
                  viewBox="0 0 24 24"
                  strokeWidth={1.5}
                  stroke="currentColor"
                  className="h-12 w-12"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M6.827 6.175A2.31 2.31 0 0 1 5.186 7.23c-.38.054-.757.112-1.134.175C2.999 7.58 2.25 8.507 2.25 9.574V18a2.25 2.25 0 0 0 2.25 2.25h15A2.25 2.25 0 0 0 21.75 18V9.574c0-1.067-.75-1.994-1.802-2.169a47.865 47.865 0 0 0-1.134-.175 2.31 2.31 0 0 1-1.64-1.055l-.822-1.316a2.192 2.192 0 0 0-1.736-1.039 48.774 48.774 0 0 0-5.232 0 2.192 2.192 0 0 0-1.736 1.039l-.821 1.316Z"
                  />
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="M16.5 12.75a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0ZM18.75 10.5h.008v.008h-.008V10.5Z"
                  />
                </svg>
                <p className="text-sm">
                  Camera is off. Click{" "}
                  <span className="text-slate-300">Start session</span> to begin
                  recognition.
                </p>
              </div>
            )}
          </div>
        </Card>

        <Card className="flex flex-col p-5">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-300">
              Recognized this session
            </h3>
            <span className="rounded-full bg-emerald-500/15 px-2.5 py-0.5 text-xs font-medium text-emerald-300">
              {state?.students.length ?? 0}
            </span>
          </div>

          {state?.students && state.students.length > 0 ? (
            <ul className="space-y-2 overflow-y-auto">
              {state.students.map((s) => (
                <li
                  key={s.student_id}
                  className="flex items-center justify-between rounded-lg border border-white/5 bg-white/[0.03] px-3 py-2"
                >
                  <div className="flex items-center gap-3">
                    <span className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-emerald-500/80 to-sky-500/80 text-xs font-bold text-white">
                      {s.name.charAt(0).toUpperCase()}
                    </span>
                    <div>
                      <p className="text-sm font-medium text-slate-100">{s.name}</p>
                      <p className="font-mono text-[11px] text-slate-400">
                        {s.student_id}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-slate-300">{s.time}</p>
                    {s.confidence !== null ? (
                      <p className="text-[11px] text-emerald-300">
                        {(s.confidence * 100).toFixed(0)}%
                      </p>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <div className="flex flex-1 items-center justify-center text-center text-sm text-slate-500">
              {running
                ? "Waiting for a recognized face… blink to verify."
                : "Start a session to see recognized students here."}
            </div>
          )}
        </Card>
      </div>

      <p className="text-xs text-slate-500">
        Tip: blink naturally — the liveness check requires a blink before marking
        attendance. Recognized names and boxes are drawn on the feed above.
      </p>
    </div>
  );
}
