"use client";

import { BookOpen, Server } from "lucide-react";
import { useState } from "react";

export default function Hero() {
  const [hovered, setHovered] = useState<"docs" | "deploy" | null>(null);

  const GITHUB_REPO_URL = "https://github.com/semwalhritvik/shift-happens";

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center px-6 pt-28 pb-32 overflow-hidden">
      {/* Ambient grid background */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          backgroundImage:
            "linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px)",
          backgroundSize: "64px 64px",
        }}
      />
      {/* Radial glow center */}
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background:
            "radial-gradient(ellipse 70% 50% at 50% 40%, rgba(0,229,255,0.04) 0%, transparent 70%)",
        }}
      />

      <div className="relative z-10 max-w-5xl mx-auto text-center flex flex-col items-center gap-8">
        {/* Status pill */}
        <div className="flex items-center gap-2 border border-white/[0.08] rounded-full px-4 py-1.5 bg-white/[0.04] backdrop-blur-md">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00E5FF] animate-pulse" style={{ boxShadow: "0 0 8px #00E5FF" }} />
          <span className="text-xs font-mono text-gray-300 tracking-widest uppercase">
            Open Source · GCP Native · Zero Egress
          </span>
        </div>

        {/* Headline */}
        <h1 className="text-6xl md:text-8xl font-semibold leading-none tracking-tighter text-white">
          Zero Latency.
          <br />
          Total Data
          <br />
          Sovereignty.
        </h1>

        {/* Sub-headline */}
        <p className="max-w-2xl text-lg md:text-xl text-gray-400 leading-relaxed font-light">
          Deploy a GCP-native, open-source ML observability pipeline within your
          VPC. Monitor drift, ensure fairness, and automate Vertex AI retraining
          without data egress.
        </p>

        {/* Hero Media Container */}
        <div className="w-full max-w-3xl mt-6 rounded-xl border border-white/[0.08] bg-white/[0.02] backdrop-blur-2xl overflow-hidden shadow-2xl">
          {/* YouTube video player */}
          <div className="relative flex items-center justify-center bg-black w-full" style={{ aspectRatio: "16/9" }}>
            <iframe
              width="100%"
              height="100%"
              src="https://www.youtube.com/embed/ztcF6HXznPk?rel=0&modestbranding=1"
              title="YouTube video player"
              frameBorder="0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
              referrerPolicy="strict-origin-when-cross-origin"
              allowFullScreen
            ></iframe>
          </div>

          {/* CTA row */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 px-8 py-6 border-t border-white/[0.06] bg-white/[0.01]">
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              onMouseEnter={() => setHovered("docs")}
              onMouseLeave={() => setHovered(null)}
              className="flex items-center gap-2 px-6 py-3 rounded-lg border border-[#333] text-sm text-gray-300 hover:text-white hover:border-[#555] transition-all duration-200 w-full sm:w-auto justify-center"
            >
              <BookOpen className="w-4 h-4" />
              View Documentation
            </a>
            <a
              href={GITHUB_REPO_URL}
              target="_blank"
              rel="noopener noreferrer"
              onMouseEnter={() => setHovered("deploy")}
              onMouseLeave={() => setHovered(null)}
              className="flex items-center gap-2 px-6 py-3 rounded-lg border text-sm font-medium transition-all duration-200 w-full sm:w-auto justify-center"
              style={{
                background: hovered === "deploy" ? "#00E5FF" : "#1a1a1a",
                borderColor: hovered === "deploy" ? "#00E5FF" : "#444",
                color: hovered === "deploy" ? "#000" : "#fff",
              }}
            >
              <Server className="w-4 h-4" />
              Deploy on GCP
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}
