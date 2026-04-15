"use client";

import { Shield, Lock, Terminal, ArrowRight } from "lucide-react";

const stakeholders = [
  {
    Icon: Shield,
    tag: "For Clients",
    title: "Total Data Privacy",
    description:
      "Your inference data, model weights, and customer records never leave your VPC boundary. Full GDPR and HIPAA posture by architecture, not policy.",
    items: ["Zero data egress", "Customer PII stays local", "Audit-ready by default"],
  },
  {
    Icon: Lock,
    tag: "For Consultancies",
    title: "Governance & IP Protection",
    description:
      "Deliver ML observability to clients without exposing their proprietary data to third-party SaaS. IP stays where it belongs.",
    items: ["No vendor lock-in", "White-label ready", "SOC 2 compliant infra"],
  },
  {
    Icon: Terminal,
    tag: "For Engineers",
    title: "GCP-Native, Zero Latency",
    description:
      "Built on Pub/Sub, BigQuery, and Vertex AI Pipelines. Drop-in SDK, zero infra setup. Observability that runs at inference speed.",
    items: ["pip install shifthappens", "< 5ms SDK overhead", "Vertex AI native"],
    mono: "Status: Operational",
  },
];

export default function Stakeholders() {
  return (
    <>
      {/* Stakeholder section */}
      <section id="enterprise" className="px-6 py-32">
        <div className="max-w-7xl mx-auto">
          <div className="mb-12 flex flex-col gap-2">
            <span className="font-mono text-xs text-[#00E5FF]/60 uppercase tracking-widest">
              Section 04
            </span>
            <h2 className="text-3xl md:text-4xl font-semibold text-white tracking-tight">
              Built for Every Stakeholder
            </h2>
            <p className="text-gray-400 text-lg max-w-xl leading-relaxed">
              One platform. Three audiences. Zero compromises on privacy.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {stakeholders.map(({ Icon, tag, title, description, items, mono }) => (
              <div
                key={tag}
                className="group rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-2xl p-7 flex flex-col gap-5 hover:border-[#00E5FF]/20 hover:bg-white/[0.05] transition-all duration-300 shadow-2xl"
              >
                {/* Icon */}
                <div className="w-11 h-11 rounded-xl border border-white/[0.08] bg-white/[0.05] flex items-center justify-center group-hover:border-[#00E5FF]/30 group-hover:bg-[#00E5FF]/[0.05] transition-all duration-300">
                  <Icon className="w-5 h-5 text-gray-300 group-hover:text-white transition-colors duration-300" strokeWidth={1.5} />
                </div>

                {/* Text */}
                <div className="flex flex-col gap-2">
                  <span className="font-mono text-[10px] text-[#00E5FF]/70 uppercase tracking-widest">{tag}</span>
                  <h3 className="text-lg font-semibold text-white tracking-tight">{title}</h3>
                  <p className="text-sm text-gray-400 leading-relaxed">{description}</p>
                </div>

                {/* Bullet list */}
                <ul className="flex flex-col gap-2 mt-auto">
                  {items.map((item) => (
                    <li key={item} className="flex items-center gap-2 font-mono text-xs text-gray-400">
                      <span className="w-1 h-1 rounded-full bg-white/30 flex-shrink-0" />
                      {item}
                    </li>
                  ))}
                </ul>

                {/* Monospace status (engineers card) */}
                {mono && (
                  <div className="pt-4 border-t border-white/[0.06] flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF] animate-pulse" style={{ boxShadow: "0 0 8px #00E5FF" }} />
                    <span className="font-mono text-xs text-[#00E5FF]">{mono}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="px-6 pb-20 pt-32 border-t border-white/[0.06]">
        <div className="max-w-7xl mx-auto flex flex-col items-center gap-16">
          {/* CTA block */}
          <div className="flex flex-col items-center gap-6 text-center">
            <h2 className="text-5xl md:text-7xl font-semibold text-white tracking-tighter leading-none max-w-2xl">
              Own Your Models.
              <br />
              Trust Your Data.
            </h2>
            <p className="text-gray-400 text-lg max-w-md leading-relaxed">
              Deploy ShiftHappens inside your VPC in under 15 minutes.
            </p>
            <a
              href="https://github.com/semwalhritvik/shift-happens"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-4 flex items-center gap-2 px-8 py-4 rounded-xl font-medium text-sm transition-all duration-300 group"
              style={{
                background: "rgba(255,255,255,0.9)",
                color: "#000",
                backdropFilter: "blur(12px)",
                border: "1px solid rgba(255,255,255,0.2)",
                boxShadow: "0 0 40px rgba(255,255,255,0.05)",
              }}
              onMouseOver={(e) => {
                (e.currentTarget as HTMLAnchorElement).style.background = "#00E5FF";
                (e.currentTarget as HTMLAnchorElement).style.boxShadow = "0 0 40px rgba(0,229,255,0.2)";
              }}
              onMouseOut={(e) => {
                (e.currentTarget as HTMLAnchorElement).style.background = "rgba(255,255,255,0.9)";
                (e.currentTarget as HTMLAnchorElement).style.boxShadow = "0 0 40px rgba(255,255,255,0.05)";
              }}
            >
              Deploy ShiftHappens
              <ArrowRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform duration-200" />
            </a>
          </div>

          {/* Divider */}
          <div className="w-full h-px bg-white/[0.06]" />

          {/* Bottom row */}
          <div className="w-full flex flex-col sm:flex-row items-center justify-between gap-4 text-xs font-mono text-gray-500">
            <span>© 2025 ShiftHappens. All rights reserved.</span>
            <div className="flex items-center gap-6">
              <span>MIT License</span>
              <a href="#" className="hover:text-gray-300 transition-colors">Privacy</a>
              <a href="https://github.com/semwalhritvik/shift-happens" target="_blank" rel="noopener noreferrer" className="hover:text-gray-300 transition-colors">GitHub</a>
              <a href="#" className="hover:text-gray-300 transition-colors">Docs</a>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
