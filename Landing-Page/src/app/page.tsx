import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Architecture from "./components/Architecture";
import MetricCards from "./components/MetricCards";
import Stakeholders from "./components/Stakeholders";

export default function Home() {
  return (
    <div className="relative min-h-screen bg-black text-white overflow-x-hidden">
      {/* Ambient background glowing orbs to make the glass effect pop */}
      <div className="fixed top-0 left-1/4 w-[40rem] h-[40rem] bg-[#00E5FF]/[0.03] rounded-full blur-[150px] pointer-events-none" />
      <div className="fixed bottom-0 right-1/4 w-[50rem] h-[50rem] bg-indigo-500/[0.03] rounded-full blur-[120px] pointer-events-none" />
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[60rem] h-[60rem] bg-[#00E5FF]/[0.02] rounded-full blur-[150px] pointer-events-none" />

      <div className="relative z-10">
        <Navbar />
        <Hero />
        <Architecture />
        <MetricCards />
        <Stakeholders />
      </div>
    </div>
  );
}
