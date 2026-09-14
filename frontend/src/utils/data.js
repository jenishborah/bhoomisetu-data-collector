export const STAGES = ["3a", "3A", "3D"];

export function asArray(value) { return Array.isArray(value) ? value : []; }
export function projectFromResponse(response) { return response?.project || response || {}; }
export function listFromResponse(response) { return asArray(response?.projects || response?.items || response); }
export function riskFromResponse(response) { return response?.risk || response || {}; }

export function getValue(record, keys, fallback = null) {
  for (const key of keys) {
    const value = record?.[key];
    if (value !== undefined && value !== null && value !== "") return value;
  }
  return fallback;
}

export function toNumber(value, fallback = null) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function formatNumber(value, fallback = "—") {
  const number = toNumber(value);
  return number === null ? fallback : new Intl.NumberFormat("en-IN").format(number);
}

export function formatPercent(value, fallback = "—") {
  const number = toNumber(value);
  if (number === null) return fallback;
  const percentage = number <= 1 ? number * 100 : number;
  return `${percentage.toFixed(1)}%`;
}

export function formatDate(value, fallback = "Not available") {
  if (!value) return fallback;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-IN", { day: "2-digit", month: "short", year: "numeric" }).format(date);
}

export function formatDateTime(value, fallback = "Not available") {
  if (!value) return fallback;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-IN", { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }).format(date);
}

export function getRiskBand(probability) {
  if (probability === null || probability === undefined) return null;
  const p = Math.max(0, Math.min(1, Number(probability)));
  if (p < 0.20) return "LOW";
  if (p < 0.40) return "MODERATE";
  if (p < 0.70) return "HIGH";
  return "CRITICAL";
}

export function getHorizon(riskResponse, horizon) {
  const risk = riskFromResponse(riskResponse);
  const item = risk?.horizons?.[horizon] || risk?.[horizon] || risk?.predictions?.[horizon] || {};
  const probability = getValue(item, ["probability", "calibrated_probability"]);
  const percentage = getValue(item, ["percentage", "risk_percentage"], probability === null ? null : Number(probability) * 100);
  const band = getValue(item, ["band", "risk_band"]);
  return { ...item, probability: toNumber(probability), percentage: toNumber(percentage), band: band || getRiskBand(probability) };
}

export function projectRisk(project) {
  const percent = getValue(project, ["risk_90d", "risk_percentage", "headline_percentage", "risk_score"]);
  const band = getValue(project, ["risk_band", "headline_band", "band"]);
  const prob = percent !== null ? Number(percent) / 100 : null;
  return { percentage: toNumber(percent), band: band || getRiskBand(prob) };
}

export function stageLabel(stage) {
  if (stage === "3a") return "3a — Preliminary notification";
  if (stage === "3A") return "3A — Declaration";
  if (stage === "3D") return "3D — Award / acquisition";
  return stage || "Not available";
}

export function riskClass(band) { return String(band || "unknown").toLowerCase().replace(/\s+/g, "-"); }
export function humanize(value) { return value ? String(value).replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase()) : "Not available"; }
export function safeText(value, fallback = "—") { return value === undefined || value === null || value === "" ? fallback : String(value); }

export function stateRows(overview, riskOverview) {
  const overviewRows = asArray(overview?.projects_by_state);
  const riskRows = asArray(riskOverview?.state_risk || riskOverview?.state_distribution);
  const riskByState = new Map(riskRows.map((row) => [row.state, row]));
  return overviewRows.map((row) => {
    const matchingRisk = riskByState.get(row.state) || {};
    return {
      ...matchingRisk,
      ...row,
      ongoing_projects: getValue(row, ["ongoing_projects", "projects"], matchingRisk.projects || 0),
      high_risk_projects: getValue(row, ["high_risk_projects", "high_risk"], matchingRisk.high_risk || 0),
      critical_projects: getValue(row, ["critical_projects", "critical"], matchingRisk.critical || 0),
      average_90d_risk: getValue(row, ["average_90d_risk", "average_risk_percentage"], matchingRisk.average_risk_percentage),
    };
  });
}

export function riskSummary(overview, riskOverview) {
  const source = overview?.risk_summary || riskOverview?.risk_summary || riskOverview?.risk_distribution || {};
  return {
    low: toNumber(source.low ?? source.LOW, 0),
    moderate: toNumber(source.moderate ?? source.MODERATE, 0),
    high: toNumber(source.high ?? source.HIGH, 0),
    critical: toNumber(source.critical ?? source.CRITICAL, 0),
  };
}
