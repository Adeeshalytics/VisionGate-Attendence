"use client";

import { useState } from "react";
import AnalyticsTab from "@/components/AnalyticsTab";
import Header from "@/components/Header";
import HistoryTab from "@/components/HistoryTab";
import LiveSessionTab from "@/components/LiveSessionTab";
import RegistryTab from "@/components/RegistryTab";
import TodayTab from "@/components/TodayTab";

const TABS = [
  { id: "today", label: "Today" },
  { id: "live", label: "Live session" },
  { id: "history", label: "History" },
  { id: "registry", label: "Registry" },
  { id: "analytics", label: "Analytics" },
] as const;

type TabId = (typeof TABS)[number]["id"];

export default function Home() {
  const [tab, setTab] = useState<TabId>("today");

  return (
    <div className="flex min-h-full flex-col">
      <Header />

      <main className="mx-auto w-full max-w-7xl flex-1 px-6 py-8">
        <nav className="mb-8 flex gap-1 rounded-xl border border-white/10 bg-white/[0.03] p-1">
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex-1 rounded-lg px-4 py-2 text-sm font-medium transition ${
                tab === t.id
                  ? "bg-gradient-to-r from-sky-500/90 to-violet-500/90 text-white shadow"
                  : "text-slate-400 hover:bg-white/5 hover:text-slate-200"
              }`}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {tab === "today" && <TodayTab />}
        {tab === "live" && <LiveSessionTab />}
        {tab === "history" && <HistoryTab />}
        {tab === "registry" && <RegistryTab />}
        {tab === "analytics" && <AnalyticsTab />}
      </main>

      <footer className="border-t border-white/10 px-6 py-4 text-center text-xs text-slate-500">
        VisionGate · University of Ruhuna · EE7204 / EC7205
      </footer>
    </div>
  );
}
