"use client";

import { useState } from "react";
import { RefreshCw } from "lucide-react";

/* ─── Shared glass card wrapper ───────────────────────────────── */
const glassCard = "rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-2xl shadow-2xl flex flex-col h-full";
const terminalBg = "rounded-lg border border-white/[0.06] bg-black/60 backdrop-blur-sm";

/* ─── Card 1: Drift Detection ─────────────────────────────────── */
function DriftCard() {
  const metrics = [
    { key: "PSI(Income)", value: "0.12", status: "Stable", ok: true },
    { key: "PSI(CreditScore)", value: "0.31", status: "Warning", ok: false },
    { key: "PSI(Age)", value: "0.05", status: "Stable", ok: true },
    { key: "KS(Income)", value: "0.09", status: "Stable", ok: true },
  ];

  return (
    <div className={glassCard}>
      <div className="px-7 pt-7 pb-5 border-b border-white/[0.06]">
        <span className="font-mono text-[10px] text-[#00E5FF]/70 uppercase tracking-widest">Module 01</span>
        <h3 className="text-xl font-semibold text-white mt-1 tracking-tight">Real-time Drift Detection</h3>
        <p className="text-sm text-gray-400 mt-1.5 leading-relaxed">
          Population Stability Index computed across every inference batch.
        </p>
      </div>
      <div className="px-7 py-6 flex flex-col gap-0 flex-1">
        {/* Terminal header */}
        <div className="flex items-center gap-2 mb-3">
          <span className="w-2 h-2 rounded-full bg-red-500/60" />
          <span className="w-2 h-2 rounded-full bg-yellow-500/60" />
          <span className="w-2 h-2 rounded-full bg-green-500/60" />
          <span className="ml-2 font-mono text-[10px] text-gray-400 tracking-wider">drift_monitor.log</span>
        </div>
        {/* Terminal */}
        <div className={`${terminalBg} p-4 flex flex-col gap-3 font-mono text-xs`}>
          {metrics.map((m) => (
            <div key={m.key} className="flex items-center justify-between gap-3">
              <span className="text-gray-400">{m.key}:</span>
              <span className="text-gray-200 font-medium">{m.value}</span>
              <div className="flex items-center gap-1.5 ml-auto">
                <span
                  className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                  style={{ background: m.ok ? "#22c55e" : "#ef4444", boxShadow: m.ok ? "0 0 6px #22c55e" : "0 0 6px #ef4444" }}
                />
                <span style={{ color: m.ok ? "#4ade80" : "#f87171" }}>
                  [{m.status}]
                </span>
              </div>
            </div>
          ))}
        </div>
        {/* Sparkline */}
        <div className="mt-5 flex items-end gap-0.5 h-10">
          {Array.from({ length: 28 }).map((_, i) => {
            const h = 30 + Math.sin(i * 0.7) * 20 + (i > 18 ? 18 : 0);
            const warn = i > 18;
            return (
              <div
                key={i}
                className="flex-1 rounded-[1px]"
                style={{
                  height: `${h}%`,
                  background: warn ? "rgba(239,68,68,0.7)" : "rgba(255,255,255,0.12)",
                }}
              />
            );
          })}
        </div>
        <div className="mt-2 flex justify-between">
          <span className="font-mono text-[10px] text-gray-500">T-28d</span>
          <span className="font-mono text-[10px] text-gray-500">Now</span>
        </div>
      </div>
    </div>
  );
}

/* ─── Card 2: Fairness ────────────────────────────────────────── */
function FairnessCard() {
  const groups = [
    { label: "Group A", value: 0.94, demo: "Age 18–35" },
    { label: "Group B", value: 0.96, demo: "Age 35–55" },
    { label: "Group C", value: 0.89, demo: "Age 55+" },
    { label: "Group D", value: 0.97, demo: "Control" },
  ];

  return (
    <div className={glassCard}>
      <div className="px-7 pt-7 pb-5 border-b border-white/[0.06]">
        <span className="font-mono text-[10px] text-[#00E5FF]/70 uppercase tracking-widest">Module 02</span>
        <h3 className="text-xl font-semibold text-white mt-1 tracking-tight">Ensuring Model Fairness</h3>
        <p className="text-sm text-gray-400 mt-1.5 leading-relaxed">
          Demographic parity and equalized odds across all cohorts.
        </p>
      </div>
      <div className="px-7 py-6 flex flex-col gap-5 flex-1">
        {/* Parity metric */}
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-4xl font-bold text-white">0.98</span>
          <div className="flex flex-col">
            <span className="font-mono text-xs text-[#00E5FF]">Parity Index</span>
            <span className="font-mono text-[10px] text-gray-400">Max disparity: 8%</span>
          </div>
        </div>
        {/* Bar chart */}
        <div className="flex flex-col gap-3">
          {groups.map((g) => (
            <div key={g.label} className="flex items-center gap-3">
              <div className="w-16 flex flex-col">
                <span className="font-mono text-xs text-gray-200 font-medium">{g.label}</span>
                <span className="font-mono text-[9px] text-gray-500">{g.demo}</span>
              </div>
              <div className="flex-1 h-5 rounded bg-white/[0.04] border border-white/[0.06] overflow-hidden">
                <div
                  className="h-full rounded"
                  style={{
                    width: `${g.value * 100}%`,
                    background: g.value >= 0.92
                      ? "linear-gradient(90deg, rgba(34,197,94,0.2), rgba(34,197,94,0.7))"
                      : "linear-gradient(90deg, rgba(239,68,68,0.2), rgba(239,68,68,0.7))",
                  }}
                />
              </div>
              <span className="font-mono text-xs text-gray-200 w-10 text-right font-medium">
                {(g.value * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
        <div className="mt-auto pt-4 border-t border-white/[0.06] font-mono text-xs text-gray-400">
          Threshold: &gt; 0.90 · Method: Demographic Parity
        </div>
      </div>
    </div>
  );
}

/* ─── Card 3: Remediation ─────────────────────────────────────── */
function RemediationCard() {
  const [triggered, setTriggered] = useState(false);
  const [step, setStep] = useState(0);

  const steps = [
    "Validating drift threshold...",
    "Fetching training dataset...",
    "Submitting Vertex AI Pipeline...",
    "Status: Pipeline Triggered ✓",
  ];

  function handleTrigger() {
    if (triggered) return;
    setTriggered(true);
    steps.forEach((_, i) => {
      setTimeout(() => setStep(i + 1), i * 600 + 200);
    });
  }

  return (
    <div className={glassCard}>
      <div className="px-7 pt-7 pb-5 border-b border-white/[0.06]">
        <span className="font-mono text-[10px] text-[#00E5FF]/70 uppercase tracking-widest">Module 03</span>
        <h3 className="text-xl font-semibold text-white mt-1 tracking-tight">Automated Remediation</h3>
        <p className="text-sm text-gray-400 mt-1.5 leading-relaxed">
          One-click retraining pipeline dispatch to Vertex AI.
        </p>
      </div>
      <div className="px-7 py-6 flex flex-col gap-6 flex-1">
        {/* Trigger button */}
        <button
          onClick={handleTrigger}
          disabled={triggered}
          className="flex items-center justify-center gap-2 w-full py-3 rounded-lg border transition-all duration-300 text-sm font-medium"
          style={{
            borderColor: triggered ? "rgba(255,255,255,0.06)" : "rgba(255,255,255,0.15)",
            color: triggered ? "rgba(255,255,255,0.3)" : "#fff",
            background: triggered ? "rgba(255,255,255,0.02)" : "rgba(255,255,255,0.06)",
            cursor: triggered ? "not-allowed" : "pointer",
            backdropFilter: "blur(8px)",
          }}
        >
          <RefreshCw
            className={`w-4 h-4 ${triggered ? "animate-spin" : ""}`}
            style={{
              color: triggered ? "rgba(0,229,255,0.5)" : "currentColor",
              animationDuration: "1.5s"
            }}
          />
          {triggered ? "Dispatching..." : "Trigger Retrain"}
        </button>

        {/* Dashed arrow */}
        <div className="flex flex-col items-center gap-0">
          <div className="w-px h-8 border-l border-dashed border-white/20" />
          <span className="text-white/30 text-xs">▼</span>
        </div>

        {/* Status terminal */}
        <div className={`flex-1 ${terminalBg} p-4 font-mono text-xs flex flex-col gap-2`}>
          {step === 0 && (
            <span className="text-gray-500">&gt; Awaiting trigger...</span>
          )}
          {steps.slice(0, step).map((s, i) => (
            <div key={i} className="flex items-start gap-2">
              <span className="text-gray-500 flex-shrink-0">&gt;</span>
              <span
                style={{
                  color: i === steps.length - 1 && step === steps.length
                    ? "#00E5FF"
                    : "rgba(255,255,255,0.7)",
                  textShadow: i === steps.length - 1 && step === steps.length
                    ? "0 0 12px rgba(0,229,255,0.4)"
                    : "none",
                }}
              >
                {s}
              </span>
            </div>
          ))}
          {step === steps.length && (
            <div className="mt-2 pt-2 border-t border-white/[0.06] flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF] animate-pulse" style={{ boxShadow: "0 0 8px #00E5FF" }} />
              <span className="font-mono text-[10px] text-[#00E5FF] tracking-widest">
                PIPELINE_ID: vx-2847-retrain
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ─── Section wrapper ─────────────────────────────────────────── */
export default function MetricCards() {
  return (
    <section id="features" className="px-6 py-32">
      <div className="max-w-7xl mx-auto">
        <div className="mb-12 flex flex-col gap-2">
          <span className="font-mono text-xs text-[#00E5FF]/60 uppercase tracking-widest">
            Section 03
          </span>
          <h2 className="text-3xl md:text-4xl font-semibold text-white tracking-tight">
            Metric Proof
          </h2>
          <p className="text-gray-400 text-lg max-w-xl leading-relaxed">
            Real-time signals, not dashboards. Every metric computed inside your VPC.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <DriftCard />
          <FairnessCard />
          <RemediationCard />
        </div>
      </div>
    </section>
  );
}
