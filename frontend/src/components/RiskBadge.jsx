import { riskClass, safeText } from "../utils/data";

export default function RiskBadge({ band, value, compact = false }) {
  const label = safeText(band, "UNSCORED").toUpperCase();
  return <span className={`risk-badge ${riskClass(label)} ${compact ? "compact" : ""}`}><span className="risk-dot" aria-hidden="true" />{label}{value !== undefined && value !== null ? ` · ${value}` : ""}</span>;
}
