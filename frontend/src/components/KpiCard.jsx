export default function KpiCard({ label, value, hint, tone = "navy" }) {
  return <article className={`kpi-card ${tone}`}><span className="kpi-label">{label}</span><strong>{value}</strong><small>{hint}</small></article>;
}
