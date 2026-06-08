"use client";

import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api";

type Phase = "running" | "done" | "timeout";

const POLL_MS = 2000;
const TIMEOUT_MS = 180_000; // give up after 3 minutes

export default function EnrollProgress({
  name,
  studentId,
  onDismiss,
}: {
  name: string;
  studentId: string;
  onDismiss: () => void;
}) {
  const [phase, setPhase] = useState<Phase>("running");
  const startRef = useRef<number>(Date.now());

  useEffect(() => {
    let active = true;
    startRef.current = Date.now();
    setPhase("running");

    const poll = setInterval(async () => {
      const age = Date.now() - startRef.current;
      try {
        const students = await api.students();
        const found = students.some(
          (s) => s.student_id.toLowerCase() === studentId.toLowerCase(),
        );
        if (found && active) {
          setPhase("done");
          clearInterval(poll);
          // Auto-dismiss shortly after success.
          setTimeout(() => active && onDismiss(), 4000);
          return;
        }
      } catch {
        // ignore transient errors while the heavy process runs
      }
      if (age > TIMEOUT_MS && active) {
        setPhase("timeout");
        clearInterval(poll);
      }
    }, POLL_MS);

    return () => {
      active = false;
      clearInterval(poll);
    };
  }, [studentId, onDismiss]);

  return (
    <div className="mx-auto max-w-7xl px-6 pb-3">
      {phase === "running" && (
        <div className="flex items-center justify-between gap-4 rounded-lg border border-amber-400/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100">
          <div className="flex items-center gap-3">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-amber-300/40 border-t-amber-200" />
            <span>
              Enrolling <strong>{name}</strong> ({studentId}) — look at the camera.
              Capturing and processing faces, this will take some time…
            </span>
          </div>
        </div>
      )}

      {phase === "done" && (
        <div className="flex items-center justify-between gap-4 rounded-lg border border-emerald-400/30 bg-emerald-500/10 px-4 py-3 text-sm text-emerald-100">
          <span>
            ✓ <strong>{name}</strong> ({studentId}) enrolled successfully and added
            to the registry.
          </span>
          <button
            onClick={onDismiss}
            className="rounded-md border border-emerald-300/30 px-2 py-1 text-xs text-emerald-100 transition hover:bg-emerald-400/10"
          >
            Dismiss
          </button>
        </div>
      )}

      {phase === "timeout" && (
        <div className="flex items-center justify-between gap-4 rounded-lg border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-100">
          <span>
            Still waiting on <strong>{name}</strong> ({studentId}). The enrollment
            window may have been closed or no face was captured. Check the webcam
            window and try again.
          </span>
          <button
            onClick={onDismiss}
            className="rounded-md border border-rose-300/30 px-2 py-1 text-xs text-rose-100 transition hover:bg-rose-400/10"
          >
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
