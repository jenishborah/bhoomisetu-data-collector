import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ErrorState, LoadingState, MethodologyNote, PanelHeader, RiskBadge } from "../components/Ui";
import { getDashboardOverview, getProjects, getRiskOverview } from "../services/api";
import { formatPercent } from "../utils/format";

function csvCell(value) {
  const text = value === null || value === undefined ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function Reports() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const load = useCallback(async () => {
    setError(null);
    try {
      const [overview, risk, projectsResponse] = await Promise.all([getDashboardOverview(), getRiskOverview(), getProjects({ limit: 1000 })]);
      setPayload({ overview, risk, projects: projectsResponse.projects || [] });
    } catch (err) { setError(err.message); }
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);
  if (!payload && !error) return <LoadingState label="Loading report center data…" />;
  if (error) return <ErrorState error={error} onRetry={load} />;

  const riskById = Object.fromEntries((payload.risk.project_risks || []).map((item) => [item.project_id, item]));
  const downloadCsv = () => {
    const columns = ["project_id", "project_name", "state", "district", "implementing_agency", "project_type", "current_stage", "days_in_current_stage", "snapshot_date", "risk_30d", "risk_60d", "risk_90d", "risk_band", "trend"];
    const rows = payload.projects.map((project) => ({ ...project, ...(riskById[project.project_id] || {}) }));
    const csv = [columns.join(","), ...rows.map((row) => columns.map((column) => csvCell(row[column])).join(","))].join("\r\n");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    link.download = `bhoomisetu-project-register-${payload.overview.last_updated || "export"}.csv`;
    document.body.appendChild(link); link.click(); link.remove(); URL.revokeObjectURL(link.href);
  };
  const priorities = payload.risk.top_high_priority_projects || [];
  const globalDrivers = payload.risk.model_explanation_summary || [];

  return <div className="page reports-page">
    <section className="page-heading"><div><span className="section-label">PROTOTYPE REPORT CENTER</span><h2>Reports &amp; Exports</h2><p>Review national and state monitoring summaries, then export the currently loaded project data as CSV.</p></div><button className="button primary" onClick={downloadCsv}>Export project register CSV</button></section>
    <section className="report-card-grid"><article className="report-card"><span>National Overview</span><strong>{payload.overview.total_projects}</strong><p>{payload.overview.ongoing_projects} ongoing projects from latest snapshots.</p></article><article className="report-card"><span>State Risk Summary</span><strong>{(payload.risk.state_risk_distribution || []).length}</strong><p>State-level calibrated-risk distributions.</p></article><article className="report-card"><span>High Priority Projects</span><strong>{priorities.length}</strong><p>HIGH or CRITICAL prototype band projects.</p></article><article className="report-card"><span>Model Explanation Summary</span><strong>{globalDrivers.length}</strong><p>Top saved global SHAP factors for the 90-day model.</p></article></section>
    <section className="panel"><PanelHeader eyebrow="NATIONAL OVERVIEW" title="Risk Summary" /><div className="risk-distribution compact">{["LOW", "MODERATE", "HIGH", "CRITICAL"].map((band) => <div className={`risk-summary-item ${band.toLowerCase()}`} key={band}><div><RiskBadge band={band} /><strong>{payload.risk.risk_distribution?.[band] ?? 0}</strong></div><small>90-day prototype band</small></div>)}</div></section>
    <section className="detail-two-column report-columns"><article className="panel"><PanelHeader eyebrow="STATE RISK SUMMARY" title="State Risk Summary" /><div className="table-scroll"><table className="data-table"><thead><tr><th>State</th><th>Projects</th><th>High</th><th>Critical</th><th>Avg. 90-day risk</th></tr></thead><tbody>{(payload.risk.state_risk_distribution || []).map((row) => <tr key={row.state}><td><strong>{row.state}</strong></td><td>{row.projects}</td><td>{row.high_risk_projects}</td><td>{row.critical_projects}</td><td>{formatPercent(row.average_90d_risk)}</td></tr>)}</tbody></table></div></article><article className="panel"><PanelHeader eyebrow="MODEL EXPLANATION SUMMARY" title="Global 90-Day SHAP Importance" detail="Global importance is not project-level or causal attribution." /><ol className="global-driver-list">{globalDrivers.map((driver) => <li key={driver.feature}><span>{driver.label}<small>{driver.category}</small></span><strong>{formatPercent(driver.normalized_importance_pct)}</strong></li>)}</ol></article></section>
    <section className="panel"><PanelHeader eyebrow="HIGH PRIORITY PROJECTS" title="Early Attention Queue" /><div className="table-scroll"><table className="data-table"><thead><tr><th>Project</th><th>State</th><th>Stage</th><th>90-day risk</th><th>Band</th><th /></tr></thead><tbody>{priorities.slice(0, 20).map((project) => <tr key={project.project_id}><td><strong>{project.project_name}</strong><small>{project.project_id}</small></td><td>{project.state}</td><td>{project.current_stage}</td><td>{formatPercent(project.risk_90d)}</td><td><RiskBadge band={project.risk_band} compact /></td><td><Link className="text-link" to={`/projects/${project.project_id}`}>Open</Link></td></tr>)}</tbody></table></div>{!priorities.length && <div className="empty-state">No HIGH or CRITICAL prototype band projects are currently present.</div>}</section>
    <section className="panel report-export-note"><PanelHeader eyebrow="EXPORT SCOPE" title="CSV Export" /><p>The CSV contains the currently loaded latest project state joined to calibrated 30/60/90-day risk values. It intentionally excludes the synthetic generator's hidden scenario field.</p></section>
    <MethodologyNote lastUpdated={payload.overview.last_updated} />
  </div>;
}

export default Reports;
