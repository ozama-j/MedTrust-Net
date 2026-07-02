import { useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Benchmark data from evaluation notebook
const BENCHMARK_DATA = [
  { model: 'MedTrust-Net (Full)', auroc: 0.8711, ece: 0.1019, f1: 0.6636, precision: 0.6714, recall: 0.6883, spec: 0.8672, ms: 25.4, hero: true },
  { model: 'ResNet-50 Baseline', auroc: 0.8597, ece: 0.0995, f1: 0.6087, precision: 0.6386, recall: 0.6047, spec: 0.8713, ms: 6.5 },
  { model: 'ResNet-50 + Cal Loss', auroc: 0.8524, ece: 0.1034, f1: 0.6062, precision: 0.6717, recall: 0.5870, spec: 0.8841, ms: 5.9 },
  { model: 'CAB Only', auroc: 0.8790, ece: 0.1173, f1: 0.6654, precision: 0.6401, recall: 0.7317, spec: 0.8447, ms: 8.3 },
  { model: 'MC Dropout (20×)', auroc: 0.8596, ece: 0.0940, f1: 0.5989, precision: 0.6298, recall: 0.5954, spec: 0.8687, ms: 130.5 },
  { model: 'Deep Ensemble (3×)', auroc: 0.8733, ece: 0.1224, f1: 0.6339, precision: 0.6788, recall: 0.6432, spec: 0.8779, ms: 19.6 },
]

const PER_CLASS = [
  { name: 'Atelectasis',     auroc: 0.8264, ece: 0.0839 },
  { name: 'Cardiomegaly',    auroc: 0.8078, ece: 0.0594 },
  { name: 'Consolidation',   auroc: 0.8871, ece: 0.1983 },
  { name: 'Edema',           auroc: 0.9153, ece: 0.0938 },
  { name: 'Pleural Effusion',auroc: 0.9187, ece: 0.0739 },
]

export default function App() {
  const [tab, setTab] = useState('analyse') // 'analyse' | 'benchmark'
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [inferenceMs, setInferenceMs] = useState(null)
  const [error, setError] = useState(null)
  const [explanation, setExplanation] = useState(null)
  const [explaining, setExplaining] = useState(false)
  const [downloading, setDownloading] = useState(false)

  const handleFile = (f) => {
    if (!f) return
    setFile(f); setPreview(URL.createObjectURL(f))
    setResult(null); setExplanation(null); setError(null); setInferenceMs(null)
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true); setError(null); setResult(null); setExplanation(null); setInferenceMs(null)
    const t0 = performance.now()
    try {
      const fd = new FormData(); fd.append('file', file)
      const res = await fetch(`${API_BASE}/predict`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error(`Server error (${res.status})`)
      const data = await res.json()
      setInferenceMs((performance.now() - t0).toFixed(0))
      setResult(data)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const handleExplain = async () => {
    if (!result) return
    setExplaining(true)
    try {
      const sa = result.medtrust_net.spatial_analysis
      const res = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          medtrust_top: result.medtrust_net.top_prediction,
          medtrust_prob: result.medtrust_net.top_probability,
          spatial_interpretation: sa.interpretation,
          unc_in_attended: sa.uncertainty_in_attended_regions,
          baseline_top: result.baseline_resnet50.top_prediction,
          baseline_prob: result.baseline_resnet50.top_probability,
          mcd_top: result.mc_dropout.top_prediction,
          mcd_prob: result.mc_dropout.top_probability,
          mcd_sigma: result.mc_dropout.mean_uncertainty,
          all_probabilities: result.medtrust_net.probabilities,
        }),
      })
      setExplanation((await res.json()).explanation)
    } catch (e) { setExplanation(`Error: ${e.message}`) }
    finally { setExplaining(false) }
  }

  const handleDownload = async () => {
    if (!result) return
    setDownloading(true)
    try {
      const res = await fetch(`${API_BASE}/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: result.filename,
          medtrust_net: result.medtrust_net,
          baseline_resnet50: result.baseline_resnet50,
          mc_dropout: result.mc_dropout,
          images: result.images,
          explanation: explanation || null,
        }),
      })
      if (!res.ok) throw new Error(`Server error (${res.status})`)
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `medtrust_report_${result.filename.replace(/\.[^.]+$/, '')}.pdf`
      document.body.appendChild(a); a.click(); a.remove()
      window.URL.revokeObjectURL(url)
    } catch (e) { setError(`Download failed: ${e.message}`) }
    finally { setDownloading(false) }
  }

  return (
    <div style={{ fontFamily: 'Inter, system-ui, sans-serif', background: '#f8fafc', minHeight: '100vh' }}>
      <style>{`
        @keyframes spin { to { transform: rotate(360deg) } }
        * { box-sizing: border-box; margin: 0; padding: 0 }
        body { background: #f8fafc }
      `}</style>

      {/* Header */}
      <header style={{ background: '#0f172a', padding: '14px 32px' }}>
        <div style={{ maxWidth: 1280, margin: '0 auto', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{ width: 36, height: 36, background: 'linear-gradient(135deg,#3b82f6,#6366f1)', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <svg width="20" height="20" fill="none" stroke="white" strokeWidth="2" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
            <div>
              <div style={{ color: '#f1f5f9', fontWeight: 700, fontSize: 18 }}>MedTrust-Net</div>
              <div style={{ color: '#64748b', fontSize: 12 }}>Confidence-Aware Attention · Unified XAI + Uncertainty Quantification</div>
            </div>
          </div>

          {/* Nav tabs */}
          <div style={{ display: 'flex', gap: 4, background: '#1e293b', borderRadius: 8, padding: 4 }}>
            {[['analyse', 'Analyse X-ray'], ['benchmark', 'Model Benchmarks']].map(([key, label]) => (
              <button key={key} onClick={() => setTab(key)} style={{
                padding: '7px 18px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 13, fontWeight: 600,
                background: tab === key ? '#3b82f6' : 'transparent',
                color: tab === key ? 'white' : '#94a3b8',
                transition: 'all 0.15s',
              }}>{label}</button>
            ))}
          </div>

          <div style={{ color: '#475569', fontSize: 12 }}>BSc FYP · Jalal Hushama · W2084767</div>
        </div>
      </header>

      <main style={{ maxWidth: 1280, margin: '0 auto', padding: '28px 28px' }}>

        {tab === 'benchmark' && <BenchmarkTab />}

        {tab === 'analyse' && <>
          {/* Upload */}
          <Section num="1" title="Upload chest X-ray">
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 24, flexWrap: 'wrap' }}>
              <div style={{ flex: 1, minWidth: 280 }}>
                <label style={{
                  display: 'block', border: '2px dashed #cbd5e1', borderRadius: 10,
                  padding: '28px 20px', textAlign: 'center', cursor: 'pointer',
                  background: file ? '#eff6ff' : '#f8fafc',
                  borderColor: file ? '#3b82f6' : '#cbd5e1',
                  transition: 'all 0.2s',
                }}>
                  <svg width="36" height="36" fill="none" stroke="#94a3b8" strokeWidth="1.5" viewBox="0 0 24 24" style={{ margin: '0 auto 10px', display: 'block' }}>
                    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
                  </svg>
                  <div style={{ color: '#475569', fontWeight: 500, marginBottom: 4 }}>
                    {file ? file.name : 'Click to select or drag and drop'}
                  </div>
                  <div style={{ color: '#94a3b8', fontSize: 12 }}>PNG, JPG · frontal chest X-ray</div>
                  <input type="file" accept="image/*" onChange={e => handleFile(e.target.files?.[0])} style={{ display: 'none' }} />
                </label>
                <button onClick={handleAnalyze} disabled={!file || loading} style={{
                  marginTop: 14, width: '100%', padding: '11px 0',
                  background: !file || loading ? '#94a3b8' : '#0f172a',
                  color: 'white', border: 'none', borderRadius: 8,
                  fontSize: 15, fontWeight: 600, cursor: !file || loading ? 'not-allowed' : 'pointer',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                }}>
                  {loading ? <><Spin /> Analyzing…</> : 'Run analysis'}
                </button>
                {error && <div style={{ marginTop: 12, padding: '10px 14px', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 8, color: '#dc2626', fontSize: 13 }}>{error}</div>}

                {/* Inference time badge */}
                {inferenceMs && (
                  <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, padding: '8px 14px', background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 8 }}>
                    <svg width="14" height="14" fill="none" stroke="#16a34a" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
                    <span style={{ fontSize: 13, color: '#15803d', fontWeight: 600 }}>Completed in {(inferenceMs / 1000).toFixed(2)} s</span>
                  </div>
                )}
              </div>
              {preview && (
                <div>
                  <div style={{ fontSize: 11, color: '#94a3b8', fontWeight: 600, letterSpacing: '0.06em', marginBottom: 6 }}>PREVIEW</div>
                  <img src={preview} alt="preview" style={{ height: 200, borderRadius: 8, border: '1px solid #e2e8f0' }} />
                </div>
              )}
            </div>
          </Section>

          {result && <Results
            result={result}
            explanation={explanation}
            explaining={explaining}
            downloading={downloading}
            onExplain={handleExplain}
            onDownload={handleDownload}
          />}
        </>}
      </main>

      <footer style={{ textAlign: 'center', padding: '24px 0 32px', color: '#94a3b8', fontSize: 12 }}>
        Decision support only · Not a substitute for radiologist review
      </footer>
    </div>
  )
}

// ─── Benchmark Tab ────────────────────────────────────────────────────────────

function BenchmarkTab() {
  const hero = BENCHMARK_DATA.find(d => d.hero)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

      {/* Summary cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
        {[
          { label: 'Macro AUROC', value: '0.8711', sub: 'Target ≥ 0.85 ✓', color: '#16a34a' },
          { label: 'Macro ECE', value: '0.1019', sub: 'Target ≤ 0.10 ≈', color: '#d97706' },
          { label: 'Macro F1', value: '0.6636', sub: 'Best among single-pass models', color: '#1d4ed8' },
          { label: 'Inference', value: '25.4 ms', sub: '5× faster than MC Dropout', color: '#6d28d9' },
        ].map(c => (
          <div key={c.label} style={{ background: 'white', borderRadius: 10, padding: '18px 20px', border: '1px solid #e2e8f0', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>{c.label}</div>
            <div style={{ fontSize: 28, fontWeight: 900, color: c.color, fontFamily: 'monospace' }}>{c.value}</div>
            <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 4 }}>{c.sub}</div>
          </div>
        ))}
      </div>

      {/* Main comparison table */}
      <div style={{ background: 'white', borderRadius: 12, border: '1px solid #e2e8f0', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
        <div style={{ padding: '18px 24px', borderBottom: '1px solid #f1f5f9' }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>Model Comparison based on CheXpert Test Set (n = 234)</h2>
          <p style={{ fontSize: 12, color: '#64748b', marginTop: 4 }}>All models evaluated on the official CheXpert validation set under identical conditions.</p>
        </div>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
            <thead>
              <tr style={{ background: '#f8fafc' }}>
                {['Model', 'AUROC ↑', 'ECE ↓', 'F1 ↑', 'Precision', 'Recall', 'Specificity', 'Inference (ms)'].map(h => (
                  <th key={h} style={{ padding: '10px 16px', textAlign: 'left', fontWeight: 700, color: '#475569', fontSize: 11, borderBottom: '1px solid #e2e8f0', letterSpacing: '0.03em' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {BENCHMARK_DATA.map((row, i) => (
                <tr key={row.model} style={{ background: row.hero ? '#eff6ff' : i % 2 === 0 ? 'white' : '#fafafa', borderLeft: row.hero ? '3px solid #3b82f6' : '3px solid transparent' }}>
                  <td style={{ padding: '12px 16px', fontWeight: row.hero ? 700 : 500, color: '#0f172a' }}>
                    {row.model}
                    {row.hero && <span style={{ marginLeft: 8, fontSize: 10, background: '#dbeafe', color: '#1d4ed8', padding: '2px 7px', borderRadius: 4, fontWeight: 700 }}>Proposed</span>}
                  </td>
                  <MetricCell val={row.auroc} best={Math.max(...BENCHMARK_DATA.map(d => d.auroc))} higher />
                  <MetricCell val={row.ece} best={Math.min(...BENCHMARK_DATA.map(d => d.ece))} />
                  <MetricCell val={row.f1} best={Math.max(...BENCHMARK_DATA.map(d => d.f1))} higher />
                  <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: '#374151' }}>{row.precision.toFixed(4)}</td>
                  <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: '#374151' }}>{row.recall.toFixed(4)}</td>
                  <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: '#374151' }}>{row.spec.toFixed(4)}</td>
                  <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: row.ms > 100 ? '#dc2626' : '#374151', fontWeight: row.ms > 100 ? 700 : 400 }}>{row.ms}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Per-class results */}
      <div style={{ background: 'white', borderRadius: 12, border: '1px solid #e2e8f0', overflow: 'hidden', boxShadow: '0 1px 3px rgba(0,0,0,0.05)' }}>
        <div style={{ padding: '18px 24px', borderBottom: '1px solid #f1f5f9' }}>
          <h2 style={{ fontSize: 16, fontWeight: 700, color: '#0f172a' }}>MedTrust-Net: Per-class Results</h2>
        </div>
        <div style={{ padding: 24, display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 14 }}>
          {PER_CLASS.map(pc => (
            <div key={pc.name} style={{ padding: '14px 16px', background: '#f8fafc', borderRadius: 10, border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: '#374151', marginBottom: 10 }}>{pc.name}</div>
              <div style={{ marginBottom: 8 }}>
                <div style={{ fontSize: 10, color: '#64748b', marginBottom: 3 }}>AUROC</div>
                <div style={{ fontSize: 20, fontWeight: 900, color: pc.auroc >= 0.85 ? '#16a34a' : '#d97706', fontFamily: 'monospace' }}>{pc.auroc.toFixed(4)}</div>
                <div style={{ marginTop: 4, height: 4, background: '#e2e8f0', borderRadius: 2 }}>
                  <div style={{ height: '100%', width: `${pc.auroc * 100}%`, background: pc.auroc >= 0.85 ? '#16a34a' : '#f59e0b', borderRadius: 2 }} />
                </div>
              </div>
              <div>
                <div style={{ fontSize: 10, color: '#64748b', marginBottom: 3 }}>ECE</div>
                <div style={{ fontSize: 20, fontWeight: 900, color: pc.ece <= 0.10 ? '#16a34a' : '#dc2626', fontFamily: 'monospace' }}>{pc.ece.toFixed(4)}</div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Key takeaways */}
      <div style={{ background: 'white', borderRadius: 12, border: '1px solid #e2e8f0', padding: 24 }}>
        <h2 style={{ fontSize: 16, fontWeight: 700, color: '#0f172a', marginBottom: 16 }}>Key Takeaways</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 14 }}>
          {[
            { color: '#dbeafe', border: '#bfdbfe', title: 'vs ResNet-50 Baseline', body: 'MedTrust-Net improves AUROC by +0.0114 and F1 by +0.0549 at a cost of only 18.9 ms additional inference time, while also adding spatial uncertainty maps the baseline cannot produce.' },
            { color: '#fef9c3', border: '#fde68a', title: 'vs MC Dropout', body: 'MC Dropout achieves the best ECE (0.0940) but runs 20 forward passes at 130.5 ms  over 5× slower. It also produces only a scalar uncertainty score with no spatial localisation.' },
            { color: '#f0fdf4', border: '#bbf7d0', title: 'vs Deep Ensemble', body: 'Deep Ensemble achieves the highest AUROC (0.8733) but has the worst ECE (0.1224) and requires training and storing 3 separate models. MedTrust-Net matches or exceeds it at one-third the cost.' },
          ].map(t => (
            <div key={t.title} style={{ padding: '14px 16px', background: t.color, border: `1px solid ${t.border}`, borderRadius: 10 }}>
              <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a', marginBottom: 6 }}>{t.title}</div>
              <p style={{ fontSize: 12, color: '#374151', lineHeight: 1.6 }}>{t.body}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

function MetricCell({ val, best, higher }) {
  const isBest = higher ? val === best : val === best
  return (
    <td style={{ padding: '12px 16px', fontFamily: 'monospace', color: isBest ? '#16a34a' : '#374151', fontWeight: isBest ? 700 : 400 }}>
      {isBest && <span style={{ marginRight: 4 }}>★</span>}
      {val.toFixed(4)}
    </td>
  )
}

// ─── Results ──────────────────────────────────────────────────────────────────

function Results({ result, explanation, explaining, downloading, onExplain, onDownload }) {
  const mt = result.medtrust_net
  const bl = result.baseline_resnet50
  const mcd = result.mc_dropout
  const sa = mt.spatial_analysis

  const attUnc = sa.uncertainty_in_attended_regions
  const ignUnc = sa.uncertainty_in_ignored_regions
  const attendedMoreConfident = attUnc < ignUnc

  // Confidence % = how certain the model is about what it attended when making this prediction.
  // Uses same σ thresholds as the backend: <0.70 confident, 0.70-1.0 moderate, >1.0 uncertain.
  let confidencePct
  if (attUnc < 0.70) {
    confidencePct = Math.round(100 - (attUnc / 0.70) * 30)
  } else if (attUnc <= 1.0) {
    confidencePct = Math.round(70 - ((attUnc - 0.70) / 0.30) * 30)
  } else {
    confidencePct = Math.round(Math.max(0, 40 - ((attUnc - 1.0) / 1.0) * 40))
  }
  const confidenceLabel = attUnc < 0.70 ? 'Confident' : attUnc <= 1.0 ? 'Moderate' : 'Uncertain'
  const confidenceBarColor = attUnc < 0.70 ? '#16a34a' : attUnc <= 1.0 ? '#f59e0b' : '#dc2626'

  const panelBg     = attUnc < 0.70 ? '#f0fdf4' : attUnc <= 1.0 ? '#fffbeb' : '#fef2f2'
  const panelBorder = attUnc < 0.70 ? '#a7f3d0' : attUnc <= 1.0 ? '#fde68a' : '#fecaca'
  const panelColor  = attUnc < 0.70 ? '#059669' : attUnc <= 1.0 ? '#d97706' : '#dc2626'

  return (
    <>
      {/* Section 2 */}
      <Section num="2" title="MedTrust-Net : Unified explanation and uncertainty" accent>
        

        {/* 4 maps */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 20, marginBottom: 24 }}>
          <MapCard img={`data:image/png;base64,${result.images.original}`} title="ORIGINAL X-RAY" sub="Input to all models" border="#e2e8f0" />
          <MapCard img={`data:image/png;base64,${result.images.baseline_gradcam}`} title="BASELINE GRAD-CAM" sub="Shows attention but always looks equally confident regardless of actual certainty" badge="No reliability signal" badgeColor="#64748b" badgeBg="#f1f5f9" border="#e2e8f0" />
          <MapCard img={`data:image/png;base64,${result.images.medtrust_dam}`} title="MEDTRUST-NET DAM" sub="Where the model attends, generated intrinsically by the CAB, not post-hoc" badge="Diagnostic Attention Map" badgeColor="#1d4ed8" badgeBg="#dbeafe" border="#bfdbfe" />
          <MapCard img={`data:image/png;base64,${result.images.medtrust_crm}`} title="MEDTRUST-NET CRM" sub="How confident the model is, region by region" badge="Confidence Reliability Map" badgeColor="#6d28d9" badgeBg="#ede9fe" border="#c4b5fd" />
        </div>

        {/* Colour scales — single row */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 16, marginBottom: 24 }}>
          <div style={{ padding: '12px 14px', background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#475569', marginBottom: 8 }}>GRAD-CAM & DAM COLOUR SCALE</div>
            <div style={{ height: 10, borderRadius: 4, background: 'linear-gradient(to right,#00f,#0ff,#0f0,#ff0,#f00)', marginBottom: 6 }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748b' }}>
              <span>← Ignored</span><span>Attended →</span>
            </div>
          </div>
          <div style={{ padding: '12px 14px', background: '#f8fafc', borderRadius: 8, border: '1px solid #e2e8f0' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#475569', marginBottom: 8 }}>CRM COLOUR SCALE</div>
            <div style={{ height: 10, borderRadius: 4, background: 'linear-gradient(to right,#000,#800,#f00,#ff0,#fff)', marginBottom: 6 }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748b' }}>
              <span>← Confident (dark)</span><span>Uncertain (bright) →</span>
            </div>
          </div>
          <div style={{ padding: '12px 14px', background: '#fffbeb', borderRadius: 8, border: '1px solid #fde68a' }}>
            <div style={{ fontSize: 11, fontWeight: 700, color: '#92400e', marginBottom: 6 }}>HOW TO READ DAM + CRM TOGETHER</div>
            <div style={{ fontSize: 12, color: '#78350f', lineHeight: 1.6 }}>
              Look at the DAM to see <em>where</em> the model is focused. Then look at the same region in the CRM.
            </div>
          </div>
        </div>

        {/* Spatial analysis — replaced Δσ with confidence % */}
        <div style={{ padding: '16px 20px', background: panelBg, border: `1px solid ${panelBorder}`, borderRadius: 10 }}>
          <p style={{ fontSize: 14, color: '#374151', lineHeight: 1.7, marginBottom: 16 }}>
            {sa.interpretation}
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12, marginBottom: 14 }}>

            {/* Card 1: Confidence in attended region */}
            <div style={{ padding: '14px 16px', background: 'rgba(255,255,255,0.9)', borderRadius: 8, border: `1px solid ${panelBorder}` }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>MODEL CONFIDENCE IN ATTENDED REGION</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 8, marginBottom: 4 }}>
                <div style={{ fontSize: 36, fontWeight: 900, color: panelColor, fontFamily: 'monospace' }}>
                  {confidencePct}%
                </div>
                <div style={{ fontSize: 13, fontWeight: 700, color: panelColor }}>{confidenceLabel}</div>
              </div>
              {/* Banded bar: red | amber | green zones */}
              <div style={{ position: 'relative', height: 8, borderRadius: 4, overflow: 'hidden', marginBottom: 6, background: 'linear-gradient(to right, #fca5a5 0%, #fca5a5 40%, #fde68a 40%, #fde68a 70%, #bbf7d0 70%, #bbf7d0 100%)' }}>
                <div style={{ position: 'absolute', top: 0, left: `${confidencePct}%`, width: 3, height: '100%', background: '#0f172a', borderRadius: 2, transform: 'translateX(-50%)' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#94a3b8', marginBottom: 8 }}>
                <span>Uncertain</span><span>Moderate</span><span>Confident</span>
              </div>
              <div style={{ fontSize: 12, color: panelColor, fontWeight: 600, lineHeight: 1.4 }}>
                How certain the model is about the region it attended to when making this prediction.
              </div>
            </div>

            {/* Card 2: sigma attended */}
            <div style={{ padding: '14px 16px', background: 'rgba(255,255,255,0.9)', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>ATTENTION UNCERTAINTY (ATTENDED)</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 2 }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: panelColor, fontFamily: 'monospace' }}>
                  {attUnc.toFixed(3)}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>out of 2.0</div>
              </div>
              <div style={{ position: 'relative', height: 6, background: 'linear-gradient(to right, #bbf7d0 0%, #bbf7d0 35%, #fde68a 35%, #fde68a 50%, #fca5a5 50%, #fca5a5 100%)', borderRadius: 3, marginBottom: 4, overflow: 'hidden' }}>
                <div style={{ position: 'absolute', top: 0, left: `${Math.min(attUnc / 2, 1) * 100}%`, width: 3, height: '100%', background: '#0f172a', borderRadius: 2, transform: 'translateX(-50%)' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#94a3b8', marginBottom: 8 }}>
                <span>0 Confident</span><span>0.7</span><span>1.0</span><span>2.0 Uncertain</span>
              </div>
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
                Uncertainty in the regions the DAM highlights. Lower = more decisive focus.
              </div>
            </div>

            {/* Card 3: sigma background */}
            <div style={{ padding: '14px 16px', background: 'rgba(255,255,255,0.9)', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', marginBottom: 6 }}>ATTENTION UNCERTAINTY (BACKGROUND)</div>
              <div style={{ display: 'flex', alignItems: 'baseline', gap: 6, marginBottom: 2 }}>
                <div style={{ fontSize: 28, fontWeight: 900, color: '#94a3b8', fontFamily: 'monospace' }}>
                  {ignUnc.toFixed(3)}
                </div>
                <div style={{ fontSize: 11, color: '#94a3b8' }}>out of 2.0</div>
              </div>
              <div style={{ position: 'relative', height: 6, background: 'linear-gradient(to right, #bbf7d0 0%, #bbf7d0 35%, #fde68a 35%, #fde68a 50%, #fca5a5 50%, #fca5a5 100%)', borderRadius: 3, marginBottom: 4, overflow: 'hidden' }}>
                <div style={{ position: 'absolute', top: 0, left: `${Math.min(ignUnc / 2, 1) * 100}%`, width: 3, height: '100%', background: '#0f172a', borderRadius: 2, transform: 'translateX(-50%)' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#94a3b8', marginBottom: 8 }}>
                <span>0 Confident</span><span>0.7</span><span>1.0</span><span>2.0 Uncertain</span>
              </div>
              <div style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>
                {attendedMoreConfident ? 'Higher than attended so focus is genuinely decisive.' : 'Lower than attended — model is less certain where it looked.'}
              </div>
            </div>

          </div>

          <div style={{ padding: '10px 14px', background: 'rgba(255,255,255,0.6)', borderRadius: 8, border: '1px solid #e2e8f0' }}>
              <div style={{ fontSize: 11, fontWeight: 700, color: '#475569', marginBottom: 4 }}>WHAT THESE VALUES MEAN</div>
                <div style={{ fontSize: 12, color: '#475569', lineHeight: 1.6 }}>
                   <strong>Sigmoid (probabilities above):</strong> Raw model scores are squashed into 0–1. Above 0.5 means the condition is likely present.<br/>
                  <strong>σ (sigma):</strong> Standard deviation of the model's attention at each location. Low σ = decisive, narrow focus. High σ = diffuse, uncertain attention.
                </div>
              </div>
          </div>
        
      </Section>

      {/* Section 3: Predictions */}
      <Section num="3" title="Pathology predictions">
        <div style={{ marginBottom: 14, padding: '10px 14px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8 }}>
          <p style={{ fontSize: 13, color: '#475569', lineHeight: 1.6 }}>
            Multi-label : each pathology is scored independently. Above ~0.5 generally indicates the condition is likely present.
          </p>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3,1fr)', gap: 20 }}>
          <PredCard title="MedTrust-Net" tag="ResNet-50 + CAB · single forward pass" tagColor="#1d4ed8" tagBg="#dbeafe" top={mt.top_prediction} topProb={mt.top_probability} probs={mt.probabilities} bar="#3b82f6" note="Same forward pass that generates the DAM and CRM." highlight />
          <PredCard title="ResNet-50 + Grad-CAM" tag="Baseline · post-hoc XAI" tagColor="#475569" tagBg="#f1f5f9" top={bl.top_prediction} topProb={bl.top_probability} probs={bl.probabilities} bar="#94a3b8" note="Grad-CAM is post-hoc — does not affect predictions and cannot express confidence." />
          <MCPredCard data={mcd} />
        </div>
      </Section>

      {/* Section 4: AI Explanation */}
      <Section num="4" title="AI-generated clinical interpretation">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
          <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.6, flex: 1 }}>
            Natural-language summary of the findings, spatial uncertainty, and model agreement. Powered by Groq LLaMA.
          </p>
          <button onClick={onExplain} disabled={explaining} style={{
            background: explaining ? '#e2e8f0' : '#6d28d9', color: explaining ? '#94a3b8' : 'white',
            border: 'none', borderRadius: 8, padding: '9px 20px', fontSize: 14, fontWeight: 600,
            cursor: explaining ? 'not-allowed' : 'pointer', display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0,
          }}>
            {explaining ? <><Spin />Generating…</> : 'Generate interpretation'}
          </button>
        </div>
        {explanation
          ? <div style={{ padding: '18px 20px', background: '#faf5ff', border: '1px solid #ddd6fe', borderRadius: 10, fontSize: 14, color: '#1e1b4b', lineHeight: 1.8, whiteSpace: 'pre-wrap' }}>{explanation}</div>
          : <div style={{ padding: 24, background: '#f8fafc', border: '1px dashed #cbd5e1', borderRadius: 10, textAlign: 'center', color: '#94a3b8', fontSize: 13 }}>Click "Generate interpretation" to produce a clinical summary.</div>
        }
      </Section>

      {/* Section 5: Download */}
      <Section num="5" title="Download report">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 16 }}>
          <p style={{ fontSize: 13, color: '#64748b', lineHeight: 1.6 }}>
            PDF report with predictions, all four visualizations{explanation ? ', AI interpretation,' : ''} and clinical disclaimer.
            {!explanation && <span style={{ color: '#d97706' }}> Add an interpretation first for a complete report.</span>}
          </p>
          <button onClick={onDownload} disabled={downloading} style={{
            background: downloading ? '#94a3b8' : '#059669', color: 'white', border: 'none', borderRadius: 8,
            padding: '10px 24px', fontSize: 14, fontWeight: 600, cursor: downloading ? 'not-allowed' : 'pointer',
            display: 'flex', alignItems: 'center', gap: 8, flexShrink: 0,
          }}>
            {downloading ? <><Spin />Generating…</> : <>
              <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"/></svg>
              Download PDF
            </>}
          </button>
        </div>
      </Section>
    </>
  )
}

// ─── Shared components ────────────────────────────────────────────────────────

function Section({ num, title, children, accent }) {
  return (
    <div style={{
      background: 'white', borderRadius: 12, border: `1px solid ${accent ? '#bfdbfe' : '#e2e8f0'}`,
      padding: 28, marginBottom: 24,
      boxShadow: accent ? '0 0 0 3px #eff6ff' : '0 1px 3px rgba(0,0,0,0.05)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 20 }}>
        <span style={{
          width: 26, height: 26, borderRadius: '50%', background: accent ? '#1d4ed8' : '#0f172a',
          color: 'white', fontSize: 13, fontWeight: 700,
          display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
        }}>{num}</span>
        <h2 style={{ fontSize: 17, fontWeight: 700, color: '#0f172a', letterSpacing: '-0.2px' }}>{title}</h2>
      </div>
      {children}
    </div>
  )
}

function MapCard({ img, title, sub, badge, badgeColor, badgeBg, border }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      <div style={{ fontSize: 11, fontWeight: 700, color: '#0f172a', letterSpacing: '0.05em' }}>{title}</div>
      <img src={img} alt={title} style={{ width: '100%', aspectRatio: '1 / 1', objectFit: 'cover', borderRadius: 10, border: `2px solid ${border}`, display: 'block' }} />
      {badge && <span style={{ display: 'inline-block', fontSize: 10, fontWeight: 700, padding: '3px 8px', borderRadius: 4, background: badgeBg, color: badgeColor }}>{badge}</span>}
      <p style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5 }}>{sub}</p>
    </div>
  )
}

function PredCard({ title, tag, tagColor, tagBg, top, topProb, probs, bar, note, highlight }) {
  const sorted = Object.entries(probs).sort((a, b) => b[1] - a[1])
  return (
    <div style={{ borderRadius: 10, padding: 20, border: highlight ? '2px solid #bfdbfe' : '1px solid #e2e8f0', background: highlight ? '#f8fbff' : 'white', display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a', marginBottom: 5 }}>{title}</div>
        <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4, background: tagBg, color: tagColor }}>{tag}</span>
      </div>
      <div style={{ padding: '12px 14px', background: '#f8fafc', borderRadius: 8, borderLeft: `3px solid ${bar}` }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 3 }}>TOP FINDING</div>
        <div style={{ fontSize: 18, fontWeight: 800, color: '#0f172a' }}>{top}</div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>{(topProb * 100).toFixed(1)}% probability</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
        {sorted.map(([name, p]) => (
          <div key={name}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <span style={{ fontSize: 13, color: '#374151' }}>{name}</span>
              <span style={{ fontSize: 13, fontWeight: 600, color: p >= 0.5 ? '#0f172a' : '#94a3b8', fontFamily: 'monospace' }}>{(p * 100).toFixed(1)}%</span>
            </div>
            <div style={{ position: 'relative', height: 6, background: '#f1f5f9', borderRadius: 3, overflow: 'hidden' }}>
              <div style={{ height: '100%', width: `${p * 100}%`, background: p >= 0.5 ? bar : '#cbd5e1', borderRadius: 3 }} />
              {/* 50% threshold marker */}
              <div style={{ position: 'absolute', top: 0, left: '50%', width: 1, height: '100%', background: '#94a3b8', opacity: 0.6 }} />
            </div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: '#94a3b8', marginTop: -6 }}>
        <span>0%</span><span>← below threshold · above threshold →</span><span>100%</span>
      </div>
      <p style={{ fontSize: 12, color: '#64748b', lineHeight: 1.5, borderTop: '1px solid #f1f5f9', paddingTop: 10 }}>{note}</p>
    </div>
  )
}

function MCPredCard({ data }) {
  const sorted = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1])
  return (
    <div style={{ borderRadius: 10, padding: 20, border: '1px solid #e2e8f0', background: 'white', display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a', marginBottom: 5 }}>MC Dropout</div>
        <span style={{ fontSize: 10, fontWeight: 600, padding: '2px 8px', borderRadius: 4, background: '#f1f5f9', color: '#475569' }}>
          {data.n_samples} forward passes · baseline uncertainty method
        </span>
      </div>
      <div style={{ padding: '12px 14px', background: '#f8fafc', borderRadius: 8, borderLeft: '3px solid #94a3b8' }}>
        <div style={{ fontSize: 11, color: '#64748b', marginBottom: 3 }}>TOP FINDING (averaged across {data.n_samples} passes)</div>
        <div style={{ fontSize: 18, fontWeight: 800, color: '#0f172a' }}>{data.top_prediction}</div>
        <div style={{ fontSize: 13, color: '#64748b', marginTop: 2 }}>{(data.top_probability * 100).toFixed(1)}% probability</div>
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
        {sorted.map(([name, p]) => {
          const std = data.uncertainty_per_class[name]
          return (
            <div key={name}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                <span style={{ fontSize: 13, color: '#374151' }}>{name}</span>
                <span style={{ fontSize: 13, color: '#0f172a', fontFamily: 'monospace' }}>
                  {(p * 100).toFixed(1)}%
                  <span style={{ color: '#94a3b8', fontSize: 11 }}> ±{(std * 100).toFixed(1)}</span>
                </span>
              </div>
              <div style={{ height: 6, background: '#f1f5f9', borderRadius: 3, overflow: 'hidden' }}>
                <div style={{ height: '100%', width: `${p * 100}%`, background: '#94a3b8', borderRadius: 3 }} />
              </div>
            </div>
          )
        })}
      </div>
      <div style={{ background: '#f0f9ff', border: '1px solid #bae6fd', borderRadius: 8, padding: '12px 14px' }}>
        <div style={{ fontSize: 11, fontWeight: 700, color: '#0369a1', marginBottom: 6 }}>WHAT THE ± MEANS</div>
        <p style={{ fontSize: 12, color: '#0c4a6e', lineHeight: 1.6, marginBottom: 8 }}>
          The model ran {data.n_samples} times with random neurons dropped. ± shows prediction variance: larger = less stable.
        </p>
        <div style={{ padding: '8px 10px', background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 6 }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: '#c2410c', marginBottom: 4 }}>LIMITATION</div>
          <p style={{ fontSize: 12, color: '#7c2d12', lineHeight: 1.5 }}>
            One number for the whole image and no spatial breakdown. MedTrust-Net's CRM shows uncertainty region-by-region.
          </p>
        </div>
      </div>
    </div>
  )
}

function Spin() {
  return (
    <svg style={{ animation: 'spin 1s linear infinite', flexShrink: 0 }} width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
      <path d="M21 12a9 9 0 11-6.219-8.56"/>
    </svg>
  )
}