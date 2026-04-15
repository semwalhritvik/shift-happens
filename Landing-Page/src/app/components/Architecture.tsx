import { Monitor, Box, Network, Database, Brain } from "lucide-react";

const nodes = [
  { icon: Monitor, label: "User App", sublabel: "Client Layer" },
  { icon: Box, label: "SDK", sublabel: "shifthappens-py" },
  { icon: Network, label: "Pub/Sub", sublabel: "GCP Broker" },
  { icon: Database, label: "BigQuery", sublabel: "Data Warehouse" },
  { icon: Brain, label: "Vertex AI", sublabel: "Retraining" },
];

export default function Architecture() {
  return (
    <section id="architecture" className="px-6 py-32">
      <div className="max-w-7xl mx-auto">
        {/* Section header */}
        <div className="mb-10 flex flex-col gap-2">
          <span className="font-mono text-xs text-gray-600 uppercase tracking-widest">
            Section 02
          </span>
          <h2 className="text-3xl md:text-4xl font-semibold text-white tracking-tight">
            The Architecture
          </h2>
        </div>

        {/* Main architecture card */}
        <div className="relative rounded-2xl border border-white/[0.08] bg-white/[0.03] backdrop-blur-2xl shadow-2xl overflow-hidden">
          {/* Card header */}
          <div className="flex items-center justify-between px-8 py-5 border-b border-white/[0.08] bg-white/[0.02]">
            <div className="flex items-center gap-3">
              {/* Traffic dots */}
              <span className="w-2.5 h-2.5 rounded-full bg-white/20" />
              <span className="w-2.5 h-2.5 rounded-full bg-white/20" />
              <span className="w-2.5 h-2.5 rounded-full bg-white/20" />
            </div>
            <div className="font-mono text-xs text-gray-400 tracking-widest text-center">
              Your GCP Environment (VPC) // ZERO EGRESS
            </div>
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF] animate-pulse" />
              <span className="font-mono text-xs text-[#00E5FF] tracking-widest uppercase">
                Live
              </span>
            </div>
          </div>

          {/* Architecture flow */}
          <div className="relative px-8 md:px-16 py-20">
            {/* ZERO EGRESS label above flow */}
            <div className="flex justify-center mb-10">
              <div className="flex items-center gap-2 px-4 py-1.5 border border-[#00E5FF]/20 rounded-full bg-[#00E5FF]/5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF]" />
                <span className="font-mono text-xs text-[#00E5FF] tracking-[0.2em] uppercase">
                  Zero Egress — Data Never Leaves Your VPC
                </span>
              </div>
            </div>

            {/* Nodes row */}
            <div className="flex items-center justify-between gap-0 overflow-x-auto">
              {nodes.map((node, i) => {
                const Icon = node.icon;
                return (
                  <div key={i} className="flex items-center flex-1 min-w-0">
                    {/* Node */}
                    <div className="flex flex-col items-center gap-3 flex-shrink-0 group">
                      <div className="relative w-16 h-16 rounded-xl border border-white/[0.08] bg-white/[0.05] flex items-center justify-center group-hover:border-[#00E5FF]/40 group-hover:bg-[#00E5FF]/[0.02] transition-all duration-300 shadow-inner">
                        {/* Node glow on hover */}
                        <div className="absolute inset-0 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"
                          style={{ background: "radial-gradient(circle at center, rgba(0,229,255,0.08) 0%, transparent 70%)" }}
                        />
                        <Icon className="w-7 h-7 text-gray-300 group-hover:text-white transition-colors duration-300" strokeWidth={1.25} />
                      </div>
                      <div className="flex flex-col items-center gap-0.5">
                        <span className="text-xs font-semibold text-gray-200 whitespace-nowrap tracking-wide">
                          {node.label}
                        </span>
                        <span className="font-mono text-[10px] text-gray-400 whitespace-nowrap">
                          {node.sublabel}
                        </span>
                      </div>
                    </div>

                    {/* Connector line (not after last node) */}
                    {i < nodes.length - 1 && (
                      <div className="flex-1 relative h-px mx-2 md:mx-4 overflow-visible">
                        {/* Static line base */}
                        <div className="absolute inset-0 bg-[#2a2a2a]" />
                        {/* Animated gradient traveling along line */}
                        <div
                          className="absolute inset-y-0 w-1/2 animate-[slide_2.5s_linear_infinite]"
                          style={{
                            background: `linear-gradient(90deg, transparent, #00E5FF, transparent)`,
                            animationDelay: `${i * 0.5}s`,
                          }}
                        />
                        {/* Arrow head */}
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-0 h-0"
                          style={{
                            borderTop: "4px solid transparent",
                            borderBottom: "4px solid transparent",
                            borderLeft: "6px solid #2a2a2a",
                          }}
                        />
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Bottom annotation */}
            <div className="mt-14 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pt-6 border-t border-white/[0.08]">
              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-px bg-white/20" />
                  <span className="font-mono text-xs text-gray-400">Internal data flow</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-px" style={{ background: "linear-gradient(90deg, transparent, #00E5FF)" }} />
                  <span className="font-mono text-xs text-[#00E5FF]/90">Active pipeline</span>
                </div>
              </div>
              <div className="font-mono text-xs text-gray-400">
                Powered by Apache Beam · BigQuery ML · Vertex AI Pipelines
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Keyframe injection via style tag */}
      <style>{`
        @keyframes slide {
          0% { left: -50%; }
          100% { left: 150%; }
        }
      `}</style>
    </section>
  );
}
