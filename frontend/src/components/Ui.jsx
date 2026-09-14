import { formatDate, formatPercent } from "../utils/format";

export function RiskBadge({ band, value, compact = false }) {
  const normalized = String(band || "UNKNOWN").toLowerCase();
  return <span className={`risk-badge ${normalized} ${compact ? "compact" : ""}`}>{band || "Unknown"}{value !== undefined && <small>{formatPercent(value)}</small>}</span>;
}

export function KpiCard({ label, value, note, tone = "navy" }) {
  return <article className={`kpi-card ${tone}`}><span>{label}</span><strong>{value}</strong><small>{note}</small></article>;
}

export function LoadingState({ label = "Loading BhoomiSetu data…" }) {
  return <section className="loading-state" aria-live="polite"><span className="loader-mark">BS</span><p>{label}</p></section>;
}

export function ErrorState({ error, onRetry }) {
  return <section className="error-state" role="alert"><div><span className="section-label">DATA CONNECTION</span><h2>Unable to load this view</h2><p>{error || "The local BhoomiSetu API did not return a response."}</p>{onRetry && <button className="button primary" onClick={onRetry}>Retry connection</button>}</div></section>;
}

export function EmptyState({ children = "No records match the current selection." }) {
  return <div className="empty-state">{children}</div>;
}

export function MethodologyNote({ lastUpdated, source = "Local synthetic project and temporal snapshot datasets" }) {
  return <aside className="methodology-note"><div><strong>Data &amp; Methodology</strong><p>Risk estimates are generated using the current prototype XGBoost models trained on synthetic temporal data. They are intended for demonstration and workflow validation, not operational decision-making.</p></div><div className="note-meta"><span>Data source</span><strong>{source}</strong>{lastUpdated && <><span>Latest snapshot</span><strong>{formatDate(lastUpdated)}</strong></>}</div></aside>;
}

export function PanelHeader({ eyebrow, title, detail, aside }) {
  return <div className="panel-header"><div>{eyebrow && <span className="section-label">{eyebrow}</span>}<h2>{title}</h2>{detail && <p>{detail}</p>}</div>{aside && <div className="panel-aside">{aside}</div>}</div>;
}
