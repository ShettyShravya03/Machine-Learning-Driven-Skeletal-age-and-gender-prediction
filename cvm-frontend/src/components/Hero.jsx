import { useNavigate } from "react-router-dom"
import { useEffect, useRef } from "react"

export default function Hero() {
  const navigate = useNavigate()
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    const resize = () => {
      canvas.width = canvas.offsetWidth
      canvas.height = canvas.offsetHeight
    }
    resize()
    window.addEventListener("resize", resize)

    const nodes = Array.from({ length: 45 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.35,
      vy: (Math.random() - 0.5) * 0.35,
      r: Math.random() * 1.8 + 0.8,
    }))

    let raf
    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      nodes.forEach((n) => {
        n.x += n.vx; n.y += n.vy
        if (n.x < 0 || n.x > canvas.width)  n.vx *= -1
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1
        ctx.beginPath()
        ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2)
        ctx.fillStyle = "rgba(56,189,248,0.55)"
        ctx.fill()
      })
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const d = Math.hypot(nodes[i].x - nodes[j].x, nodes[i].y - nodes[j].y)
          if (d < 110) {
            ctx.beginPath()
            ctx.moveTo(nodes[i].x, nodes[i].y)
            ctx.lineTo(nodes[j].x, nodes[j].y)
            ctx.strokeStyle = `rgba(56,189,248,${0.10 * (1 - d / 110)})`
            ctx.lineWidth = 0.7
            ctx.stroke()
          }
        }
      }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => { cancelAnimationFrame(raf); window.removeEventListener("resize", resize) }
  }, [])

  return (
    <section className="relative min-h-screen bg-[#030b15] flex flex-col items-center justify-center text-center px-6 overflow-hidden">
      <canvas ref={canvasRef} className="absolute inset-0 w-full h-full opacity-70 pointer-events-none" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_65%_55%_at_50%_42%,rgba(14,165,233,0.10),transparent)] pointer-events-none" />

      {/* Decorative vertebrae silhouette */}
      <div className="absolute right-[8%] top-1/2 -translate-y-1/2 opacity-[0.055] pointer-events-none select-none hidden xl:block">
        <svg width="100" height="360" viewBox="0 0 100 360" fill="none">
          {[0,1,2,3,4,5,6].map((i) => (
            <g key={i} transform={`translate(0,${i * 50})`}>
              <rect x="22" y="4" width="56" height="34" rx="7" fill="white"/>
              <rect x="6"  y="12" width="16" height="18" rx="4" fill="white"/>
              <rect x="78" y="12" width="16" height="18" rx="4" fill="white"/>
              <rect x="44" y="38" width="12" height="9"  rx="2" fill="white"/>
            </g>
          ))}
        </svg>
      </div>

      <div className="relative z-10 max-w-3xl mx-auto flex flex-col items-center">
        <span className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full border border-sky-500/25 bg-sky-500/10 text-sky-400 text-[11px] font-bold tracking-[0.18em] uppercase mb-8">
          <span className="w-1.5 h-1.5 rounded-full bg-sky-400 animate-pulse" />
          Medical AI · Deep Learning
        </span>

        <h1
          className="text-6xl md:text-[82px] font-black tracking-tight text-white leading-[0.92] mb-3"
          style={{ fontFamily: "'Syne', sans-serif" }}
        >
          Cervical
        </h1>
        <h1
          className="text-6xl md:text-[82px] font-black tracking-tight leading-[0.92] mb-3"
          style={{
            fontFamily: "'Syne', sans-serif",
            WebkitTextStroke: "1.5px rgba(56,189,248,0.7)",
            color: "transparent",
          }}
        >
          Vertebral
        </h1>
        <h1
          className="text-6xl md:text-[82px] font-black tracking-tight text-sky-400 leading-[0.92] mb-8"
          style={{ fontFamily: "'Syne', sans-serif" }}
        >
          Maturation
        </h1>

        <p className="text-slate-400 text-lg max-w-xl leading-relaxed mb-10"
          style={{ fontFamily: "'DM Sans', sans-serif" }}>
          Upload a lateral cephalometric X-ray. Our Attention U-Net segments C2–C4 vertebrae
          and the model predicts skeletal age and gender in seconds.
        </p>

        <div className="flex flex-col sm:flex-row gap-4 items-center">
          <button
            onClick={() => navigate("/analysis")}
            className="px-8 py-3.5 rounded-xl font-semibold text-white text-sm tracking-wide transition-all duration-200 hover:scale-105 hover:shadow-[0_0_28px_rgba(56,189,248,0.35)]"
            style={{
              background: "linear-gradient(135deg, #0ea5e9, #06b6d4)",
              fontFamily: "'DM Sans', sans-serif",
            }}
          >
            Start Analysis →
          </button>
          <button
            onClick={() => document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" })}
            className="px-8 py-3.5 rounded-xl font-semibold text-slate-300 text-sm tracking-wide border border-slate-700 hover:border-sky-500/50 hover:text-sky-300 transition-all duration-200"
            style={{ fontFamily: "'DM Sans', sans-serif" }}
          >
            How it works
          </button>
        </div>

        {/* Stats row */}
        <div className="mt-16 grid grid-cols-4 gap-8 border-t border-slate-800 pt-10 w-full max-w-4xl ">
          {[
            { val: "U-Net", label: "Segmentation" },
            { val: "C2–C4", label: "Vertebrae" },
            { val: "XGBOOST", label: "Age Prediction" },
            { val: "Random Forest", label: "Gender Prediction" },
          ].map((s) => (
            <div key={s.label} className="text-center">
              <p className="text-sky-400 text-xl font-black whitespace-nowrap" style={{ fontFamily: "'Syne', sans-serif" }}>{s.val}</p>
              <p className="text-slate-500 text-xs mt-0.5 tracking-wide uppercase" style={{ fontFamily: "'DM Sans', sans-serif" }}>{s.label}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
