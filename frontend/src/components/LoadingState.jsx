export default function LoadingState({ label = "Loading BhoomiSetu data…", compact = false }) {
  return <div className={compact ? "loading-state compact" : "loading-state"} role="status" aria-live="polite"><span className="loading-emblem" aria-hidden="true">BS</span><span>{label}</span></div>;
}
