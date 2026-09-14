import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { ErrorState, LoadingState, MethodologyNote, PanelHeader, RiskBadge } from "../components/Ui";
import { getDashboardOverview, getProjects, getRiskOverview } from "../services/api";
import { formatPercent } from "../utils/format";

function densityClass(row) {
  if ((row?.critical_projects || 0) > 0) return "critical";
  if ((row?.high_risk_projects || 0) > 0) return "high";
  if ((row?.average_90d_risk || 0) >= 1) return "moderate";
  return "low";
}

function GISMap() {
  const [payload, setPayload] = useState(null);
  const [selectedState, setSelectedState] = useState("");
  const [error, setError] = useState(null);
  const load = useCallback(async () => {
    setError(null);
    try {
      const [overview, risk, projectsResponse] = await Promise.all([getDashboardOverview(), getRiskOverview(), getProjects({ limit: 1000 })]);
      setPayload({ overview, risk, projects: projectsResponse.projects || [] });
      setSelectedState((current) => current || overview.projects_by_state?.[0]?.state || "");
    } catch (err) { setError(err.message); }
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);
  if (!payload && !error) return <LoadingState label="Loading state-level geographic distribution…" />;
  if (error) return <ErrorState error={error} onRetry={load} />;

  const rows = payload.overview.projects_by_state || [];
  const riskRows = payload.risk.state_risk_distribution || payload.risk.state_risk || [];
  const selected = rows.find((row) => row.state === selectedState) || rows[0];
  const stateRisk = riskRows.find((row) => row.state === selected?.state) || {};
  const riskById = Object.fromEntries((payload.risk.project_risks || []).map((item) => [item.project_id, item]));
  const stateProjects = payload.projects.filter((project) => project.state === selected?.state).sort((a, b) => (Number(riskById[b.project_id]?.risk_90d) || 0) - (Number(riskById[a.project_id]?.risk_90d) || 0));
  const stages = Object.fromEntries(["3a", "3A", "3D"].map((stage) => [stage, stateProjects.filter((project) => project.current_stage === stage).length]));

  return <div className="page gis-page">
    <section className="page-heading"><div><span className="section-label">GIS INFORMATION VIEW</span><h2>State-level Project Distribution</h2><p>State-level prototype visualization based on master-state attribution. No parcel or fabricated project coordinates are shown.</p></div><div className="data-status"><span>Visualization scope</span><strong>State-level prototype</strong></div></section>
    <section className="gis-layout">
      <article className="panel map-panel"><PanelHeader eyebrow="THEMATIC STATE VIEW" title="Project and Risk Density" detail="Select a state to inspect its project distribution." />
        <div className="map-legend"><span><i className="low" /> Lower density</span><span><i className="moderate" /> Moderate density</span><span><i className="high" /> High-priority present</span><span><i className="critical" /> Critical present</span></div>
        <div className="state-map-grid" role="list" aria-label="State-level project distribution">{rows.map((row) => <button key={row.state} className={`map-state ${densityClass(row)} ${row.state === selected?.state ? "selected" : ""}`} onClick={() => setSelectedState(row.state)} role="listitem"><strong>{row.state}</strong><span>{row.ongoing_projects} ongoing</span><small>{formatPercent(row.average_90d_risk)} avg. 90d risk</small></button>)}</div>
        <p className="map-disclaimer">This is a state-level thematic visualization. It is not a cadastral, parcel, boundary, or statutory GIS map.</p>
      </article>
      <aside className="panel gis-selection"><PanelHeader eyebrow="SELECTED STATE" title={selected?.state || "No state selected"} />
        <dl className="state-metrics"><div><dt>Ongoing projects</dt><dd>{selected?.ongoing_projects ?? 0}</dd></div><div><dt>High risk</dt><dd>{selected?.high_risk_projects ?? stateRisk.high_risk_projects ?? 0}</dd></div><div><dt>Critical</dt><dd>{selected?.critical_projects ?? stateRisk.critical_projects ?? 0}</dd></div><div><dt>Average 90-day risk</dt><dd>{formatPercent(selected?.average_90d_risk ?? stateRisk.average_90d_risk)}</dd></div></dl>
        <h3>Stage distribution</h3><div className="mini-stage-list">{Object.entries(stages).map(([stage, count]) => <div key={stage}><span>{stage}</span><strong>{count}</strong></div>)}</div>
        <p className="small-note">State totals use the latest snapshot for each project.</p>
      </aside>
    </section>
    <section className="panel selected-projects-panel"><PanelHeader eyebrow="STATE PROJECT QUEUE" title={`Projects in ${selected?.state || "selected state"}`} detail="Ordered by calibrated 90-day prototype estimate." />
      <div className="table-scroll"><table className="data-table"><thead><tr><th>Project</th><th>District</th><th>Stage</th><th>90-Day Risk</th><th>Band</th><th /></tr></thead><tbody>{stateProjects.slice(0, 20).map((project) => { const risk = riskById[project.project_id] || {}; return <tr key={project.project_id}><td><strong>{project.project_name}</strong><small>{project.project_id}</small></td><td>{project.district}</td><td><span className="stage-pill">{project.current_stage}</span></td><td>{formatPercent(risk.risk_90d)}</td><td>{risk.risk_band ? <RiskBadge band={risk.risk_band} compact /> : "—"}</td><td><Link className="text-link" to={`/projects/${project.project_id}`}>Open</Link></td></tr>; })}</tbody></table></div>
    </section>
    <MethodologyNote lastUpdated={payload.overview.last_updated} />
  </div>;
}

export default GISMap;
