import { Cpu, ScanLine, TrendingUp } from "lucide-react"

function Card({ icon, title, desc, accent }) {
  return (
    <div
      className="relative rounded-2xl p-6 border border-slate-800 bg-[#070f1c] overflow-hidden group transition-all duration-300 hover:-translate-y-1 hover:border-sky-500/30"
      style={{ fontFamily: "'DM Sans', sans-serif" }}
    >
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"
        style={{ background: "radial-gradient(circle at 30% 30%, rgba(14,165,233,0.06), transparent 65%)" }}
      />
      <div className={`mb-4 w-10 h-10 rounded-xl flex items-center justify-center ${accent}`}>
        {icon}
      </div>
      <h3 className="text-white font-bold text-base mb-2" style={{ fontFamily: "'Syne', sans-serif" }}>{title}</h3>
      <p className="text-slate-400 text-sm leading-relaxed">{desc}</p>
    </div>
  )
}

export default function ValueSection() {
  return (
    <section className="bg-[#030b15] px-6 py-16">
      <div className="max-w-5xl mx-auto">
        <p className="text-center text-xs font-bold tracking-[0.18em] uppercase text-sky-500 mb-3">
          What we do
        </p>
        <h2 className="text-center text-3xl font-black text-white mb-10"
          style={{ fontFamily: "'Syne', sans-serif" }}>
          End-to-End Automated Pipeline
        </h2>
        <div className="grid sm:grid-cols-3 gap-5">
          <Card
            icon={<ScanLine size={20} className="text-sky-400" />}
            accent="bg-sky-500/10"
            title="Deep Segmentation"
            desc="Attention U-Net architecture trained on lateral cephalograms precisely identifies C2, C3, and C4 vertebrae."
          />
          <Card
            icon={<Cpu size={20} className="text-teal-400" />}
            accent="bg-teal-500/10"
            title="Morphological Features"
            desc="79 geometric and shape features extracted per image — area, circularity, solidity, aspect ratio and more."
          />
          <Card
            icon={<TrendingUp size={20} className="text-indigo-400" />}
            accent="bg-indigo-500/10"
            title="Age & Gender Prediction"
            desc="XGBoost predicts skeletal age from vertebral morphology, while Random Forest classifies patient gender with high accuracy."
          />
        </div>
      </div>
    </section>
  )
}
