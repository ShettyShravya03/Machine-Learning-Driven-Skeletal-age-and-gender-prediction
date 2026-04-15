import { useState, useRef, useCallback } from "react"
import { useNavigate } from "react-router-dom"
import { UploadCloud, ArrowLeft, CheckCircle, AlertCircle, Loader2, ScanLine } from "lucide-react"

const API = "http://127.0.0.1:8000"

const FONT_DISPLAY = "'Syne', sans-serif"
const FONT_BODY    = "'DM Sans', sans-serif"

/* ── small reusable label ── */
function Tag({ children }) {
  return (
    <span className="text-[10px] font-bold tracking-[0.15em] uppercase text-sky-500"
      style={{ fontFamily: FONT_BODY }}>{children}</span>
  )
}

/* ── vertebrae feature row ── */
function VertRow({ v }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-800 last:border-0">
      <span className="text-sky-400 font-black text-sm w-8" style={{ fontFamily: FONT_DISPLAY }}>{v.name}</span>
      <div className="flex gap-6 text-xs text-slate-400" style={{ fontFamily: FONT_BODY }}>
        <span>Area <span className="text-slate-200">{v.area}</span></span>
        <span>AR <span className="text-slate-200">{v.aspect_ratio}</span></span>
        <span>Circ <span className="text-slate-200">{v.circularity}</span></span>
        <span>Sol <span className="text-slate-200">{v.solidity}</span></span>
      </div>
    </div>
  )
}

/* ── image panel ── */
function ImgPanel({ src, label }) {
  return (
    <div className="flex flex-col gap-2">
      <Tag>{label}</Tag>
      <div className="rounded-xl overflow-hidden border border-slate-800 bg-black">
        <img src={src} alt={label} className="w-full object-contain max-h-56" />
      </div>
    </div>
  )
}

export default function Analysis() {
  const navigate = useNavigate()
  const inputRef  = useRef(null)
  const dropRef   = useRef(null)

  const [file,       setFile]       = useState(null)
  const [preview,    setPreview]    = useState(null)
  const [dragging,   setDragging]   = useState(false)
  const [loading,    setLoading]    = useState(false)
  const [result,     setResult]     = useState(null)   // { predicted_age, vertebrae, images }
  const [error,      setError]      = useState(null)
  const [activeTab,  setActiveTab]  = useState("overlay")

  const pickFile = (f) => {
    if (!f) return
    setFile(f)
    setResult(null)
    setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files[0]
    if (f && f.type.startsWith("image/")) pickFile(f)
  }, [])

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const fd = new FormData()
    fd.append("file", file)

    try {
      const res  = await fetch(`${API}/predict`, { method: "POST", body: fd })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || "Server error")
      setResult(data)
      setActiveTab("overlay")
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const reset = () => {
    setFile(null); setPreview(null); setResult(null); setError(null)
  }

  const tabs = [
    { key: "overlay", label: "Overlay" },
    { key: "mask",    label: "Mask" },
    { key: "original",label: "Original" },
  ]

  return (
    <div className="min-h-screen bg-[#030b15] text-white" style={{ fontFamily: FONT_BODY }}>

      {/* Top bar */}
      <div className="border-b border-slate-800 px-6 py-4 flex items-center justify-between">
        <button
          onClick={() => navigate("/")}
          className="flex items-center gap-2 text-slate-400 hover:text-sky-400 transition-colors text-sm"
        >
          <ArrowLeft size={16} />
          <span>Back</span>
        </button>
        <div className="flex items-center gap-2">
          <ScanLine size={18} className="text-sky-400" />
          <span className="font-black text-sm tracking-wide" style={{ fontFamily: FONT_DISPLAY }}>
            CVM Analysis
          </span>
        </div>
        <div className="w-20" />
      </div>

      <div className="max-w-6xl mx-auto px-6 py-10">
        <div className="grid lg:grid-cols-2 gap-8">

          {/* ── LEFT: Upload ── */}
          <div className="flex flex-col gap-6">
            <div>
              <h2 className="text-2xl font-black mb-1" style={{ fontFamily: FONT_DISPLAY }}>
                Upload Radiograph
              </h2>
              <p className="text-slate-500 text-sm">Lateral cephalometric X-ray (PNG or JPG)</p>
            </div>

            {/* Drop zone */}
            <div
              ref={dropRef}
              onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
              onDragLeave={() => setDragging(false)}
              onDrop={onDrop}
              onClick={() => !file && inputRef.current?.click()}
              className={`relative rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer overflow-hidden
                ${dragging ? "border-sky-400 bg-sky-500/5" : "border-slate-700 hover:border-sky-500/50 bg-[#060e1b]"}
                ${file ? "cursor-default" : ""}`}
              style={{ minHeight: 260 }}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => pickFile(e.target.files[0])}
              />

              {preview ? (
                <div className="relative">
                  <img src={preview} alt="preview" className="w-full object-contain max-h-64" />
                  <button
                    onClick={(e) => { e.stopPropagation(); reset() }}
                    className="absolute top-3 right-3 bg-slate-900/80 hover:bg-red-500/80 text-white text-xs px-3 py-1 rounded-lg transition-colors"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center gap-3 py-16 px-8 text-center">
                  <div className="w-14 h-14 rounded-2xl bg-sky-500/10 flex items-center justify-center">
                    <UploadCloud size={26} className="text-sky-400" />
                  </div>
                  <p className="text-white font-semibold text-sm">Drag & drop or click to upload</p>
                  <p className="text-slate-500 text-xs">PNG · JPG · JPEG</p>
                </div>
              )}
            </div>

            {/* Analyze button */}
            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="w-full py-3.5 rounded-xl font-semibold text-sm tracking-wide transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed hover:scale-[1.02] hover:shadow-[0_0_24px_rgba(56,189,248,0.3)] flex items-center justify-center gap-2"
              style={{
                background: !file || loading ? "#1e293b" : "linear-gradient(135deg,#0ea5e9,#06b6d4)",
                fontFamily: FONT_DISPLAY,
              }}
            >
              {loading
                ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</>
                : <><ScanLine size={16} /> Analyze</>
              }
            </button>

            {/* Error */}
            {error && (
              <div className="flex items-start gap-3 rounded-xl bg-red-500/10 border border-red-500/20 p-4 text-sm text-red-400">
                <AlertCircle size={16} className="mt-0.5 shrink-0" />
                {error}
              </div>
            )}

            {/* Vertebrae table */}
            {result && (
              <div className="rounded-2xl border border-slate-800 bg-[#060e1b] p-5">
                <div className="flex items-center gap-2 mb-4">
                  <CheckCircle size={15} className="text-sky-400" />
                  <Tag>Extracted Features</Tag>
                </div>
                {result.vertebrae.map((v) => <VertRow key={v.name} v={v} />)}
              </div>
            )}
          </div>

          {/* ── RIGHT: Results ── */}
          <div className="flex flex-col gap-6">
            <div>
              <h2 className="text-2xl font-black mb-1" style={{ fontFamily: FONT_DISPLAY }}>
                Analysis Results
              </h2>
              <p className="text-slate-500 text-sm">Segmentation output with age and gender prediction</p>
            </div>

            {/* Age card */}
            <div className={`rounded-2xl border p-8 text-center transition-all duration-500
              ${result
                ? "border-sky-500/30 bg-gradient-to-br from-sky-500/10 to-teal-500/5"
                : "border-slate-800 bg-[#060e1b]"}`}
            >
              {!result && !loading && (
                <div className="py-8 text-slate-600">
                  <ScanLine size={36} className="mx-auto mb-3 opacity-40" />
                  <p className="text-sm">Upload an image and click Analyze</p>
                </div>
              )}
              {loading && (
                <div className="py-8 flex flex-col items-center gap-3 text-slate-400">
                  <Loader2 size={30} className="animate-spin text-sky-400" />
                  <p className="text-sm">Running Attention U-Net segmentation…</p>
                  <p className="text-xs text-slate-600">Extracting features · Predicting age</p>
                </div>
              )}
              {result && (
                <>
                  <p className="text-slate-400 text-xs uppercase tracking-widest mb-3">
                    Predicted Skeletal Age
                  </p>
                  <div className="flex items-end justify-center gap-2">
                    <span
                      className="text-7xl font-black text-sky-400"
                      style={{ fontFamily: FONT_DISPLAY, lineHeight: 1 }}
                    >
                      {result.predicted_age}
                    </span>
                    <span className="text-slate-400 text-xl pb-2">yrs</span>
                  </div>
                  {/* Gender */}
                  <div className="flex items-center justify-center gap-2 mt-5">
                    <span
                      className={`text-2xl font-black`}
                      style={{ fontFamily: FONT_DISPLAY, color: result.predicted_gender === "Female" ? "#f472b6" : "#60a5fa" }}
                    >
                      {result.predicted_gender === "Female" ? "♀" : "♂"}
                    </span>
                    <span
                      className="text-lg font-black"
                      style={{ fontFamily: FONT_DISPLAY, color: result.predicted_gender === "Female" ? "#f472b6" : "#60a5fa" }}
                    >
                      {result.predicted_gender}
                    </span>
                    <span className="text-xs text-slate-500">
                      {Math.round(result.gender_confidence * 100)}% conf.
                    </span>
                  </div>
                  <p className="text-slate-600 text-xs mt-3">
                    Based on C2, C3, C4 vertebral morphology
                  </p>
                </>
              )}
            </div>

            {/* Image tabs */}
            {result && (
              <div className="rounded-2xl border border-slate-800 bg-[#060e1b] p-5 flex flex-col gap-4">
                <div className="flex gap-1 bg-slate-900 rounded-xl p-1">
                  {tabs.map((t) => (
                    <button
                      key={t.key}
                      onClick={() => setActiveTab(t.key)}
                      className={`flex-1 py-1.5 rounded-lg text-xs font-semibold tracking-wide transition-all duration-150
                        ${activeTab === t.key
                          ? "bg-sky-500 text-white"
                          : "text-slate-500 hover:text-slate-300"}`}
                      style={{ fontFamily: FONT_DISPLAY }}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
                <div className="rounded-xl overflow-hidden border border-slate-800 bg-black">
                  <img
                    src={result.images[activeTab]}
                    alt={activeTab}
                    className="w-full object-contain max-h-72"
                  />
                </div>
                <p className="text-xs text-slate-600 text-center">
                  {activeTab === "overlay"  && "Original X-ray with vertebrae highlighted in blue"}
                  {activeTab === "mask"     && "Binary segmentation mask (C2, C3, C4)"}
                  {activeTab === "original" && "Preprocessed grayscale input to U-Net"}
                </p>
              </div>
            )}
          </div>

        </div>
      </div>
    </div>
  )
}
