"use client";

import { useEffect, useState } from "react";

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? "bg-black/80 backdrop-blur-md border-b border-[#222]"
          : "bg-transparent"
      }`}
    >
      <div className="max-w-7xl mx-auto px-8 py-5 flex items-center justify-between">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">
            <polygon points="14,2 26,8 26,20 14,26 2,20 2,8" stroke="white" strokeWidth="1.5" fill="none"/>
            <polygon points="14,2 26,8 14,14 2,8" stroke="white" strokeWidth="1" fill="white" fillOpacity="0.08"/>
            <line x1="14" y1="14" x2="14" y2="26" stroke="white" strokeWidth="1" strokeOpacity="0.5"/>
          </svg>
          <span className="text-white font-semibold text-base tracking-tight">ShiftHappens</span>
        </div>

        {/* Links */}
        <div className="hidden md:flex items-center gap-8">
          {[
            { name: "How it Works", href: "#architecture" },
            { name: "Features", href: "#features" },
            { name: "GitHub", href: "https://github.com/semwalhritvik/shift-happens" },
            { name: "Enterprise", href: "#enterprise" },
          ].map((link) => (
            <a
              key={link.name}
              href={link.href}
              className="text-sm text-gray-500 hover:text-gray-200 transition-colors duration-200 tracking-wide"
              target={link.href.startsWith("http") ? "_blank" : undefined}
            >
              {link.name}
            </a>
          ))}
        </div>
      </div>
    </nav>
  );
}
