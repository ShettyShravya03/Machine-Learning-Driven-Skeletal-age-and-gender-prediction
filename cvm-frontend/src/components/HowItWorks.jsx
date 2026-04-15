export default function HowItWorks() {
  const steps = [
    { num: "01", title: "Upload X-Ray",       desc: "Provide a lateral cephalometric radiograph (PNG or JPG)." },
    { num: "02", title: "Segmentation",        desc: "Attention U-Net identifies C2, C3, C4 vertebrae and generates a binary mask." },
    { num: "03", title: "Feature Extraction",  desc: "79 morphological features extracted: area, perimeter, circularity, solidity, etc." },
    { num: "04", title: "Age & GenderPrediction",      desc: "Trained SVR model returns the predicted skeletal age in years." },

  ]

  return (
    <section id="how-it-works" className="bg-[#030b15] px-6 pb-20">
      <div className="max-w-5xl mx-auto border border-slate-800 rounded-3xl py-14 px-10 bg-[#060e1b]">
        <p className="text-center text-xs font-bold tracking-[0.18em] uppercase text-sky-500 mb-3">
          Pipeline
        </p>
        <h2 className="text-center text-3xl font-black text-white mb-14"
          style={{ fontFamily: "'Syne', sans-serif" }}>
          How It Works
        </h2>

        <div className="grid md:grid-cols-4 gap-0">
          {steps.map((step, i) => (
            <div key={i} className="relative flex flex-col items-center text-center px-4">
              {/* Connector line */}
              {i < steps.length - 1 && (
                <div className="hidden md:block absolute top-6 left-1/2 w-full h-px bg-gradient-to-r from-sky-500/40 to-transparent" />
              )}
              <div className="relative z-10 w-12 h-12 rounded-full border border-sky-500/40 bg-[#030b15] flex items-center justify-center mb-4">
                <span className="text-sky-400 text-xs font-black tracking-wider"
                  style={{ fontFamily: "'DM Sans', sans-serif" }}>{step.num}</span>
              </div>
              <h3 className="text-white font-bold text-sm mb-2"
                style={{ fontFamily: "'Syne', sans-serif" }}>{step.title}</h3>
              <p className="text-slate-500 text-xs leading-relaxed"
                style={{ fontFamily: "'DM Sans', sans-serif" }}>{step.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}