// Small presentational primitives shared across the dashboard.

import type { ReactNode } from "react";

export function Card({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-2xl border border-white/10 bg-white/[0.03] shadow-lg shadow-black/20 ${className}`}
    >
      {children}
    </div>
  );
}

export function StatCard({
  label,
  value,
  accent = "text-sky-300",
  hint,
}: {
  label: string;
  value: ReactNode;
  accent?: string;
  hint?: string;
}) {
  return (
    <Card className="p-5">
      <p className="text-xs font-medium uppercase tracking-wider text-slate-400">
        {label}
      </p>
      <p className={`mt-2 text-3xl font-semibold ${accent}`}>{value}</p>
      {hint ? <p className="mt-1 text-xs text-slate-500">{hint}</p> : null}
    </Card>
  );
}

export function Badge({
  children,
  tone = "slate",
}: {
  children: ReactNode;
  tone?: "slate" | "green" | "red" | "amber" | "sky";
}) {
  const tones: Record<string, string> = {
    slate: "bg-slate-500/15 text-slate-300 ring-slate-400/20",
    green: "bg-emerald-500/15 text-emerald-300 ring-emerald-400/20",
    red: "bg-rose-500/15 text-rose-300 ring-rose-400/20",
    amber: "bg-amber-500/15 text-amber-300 ring-amber-400/20",
    sky: "bg-sky-500/15 text-sky-300 ring-sky-400/20",
  };
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ring-inset ${tones[tone]}`}
    >
      {children}
    </span>
  );
}

export function Spinner({ label }: { label?: string }) {
  return (
    <div className="flex items-center gap-3 text-slate-400">
      <span className="h-4 w-4 animate-spin rounded-full border-2 border-slate-500 border-t-sky-400" />
      {label ? <span className="text-sm">{label}</span> : null}
    </div>
  );
}

export function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
      {message}
    </div>
  );
}

export function EmptyState({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-dashed border-white/10 bg-white/[0.02] px-4 py-10 text-center text-sm text-slate-400">
      {message}
    </div>
  );
}

export function confidenceTone(c: number | null): "green" | "amber" | "slate" {
  if (c === null) return "slate";
  if (c >= 0.7) return "green";
  if (c >= 0.5) return "amber";
  return "slate";
}
