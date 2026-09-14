import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import StageFlow from "../components/StageFlow";
import { ErrorState, LoadingState, MethodologyNote, PanelHeader, RiskBadge } from "../components/Ui";
import { getDashboardOverview, getProjects, getRiskOverview } from "../services/api";
import { formatPercent } from "../utils/format";

function StageSentinelPage() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const [stateFilter, setStateFilter] = useState("");
  const [stageFilter, setStageFilter] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
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
  if (!payload && !error) return <LoadingState label="Loading national stage-sentinel data…" />;
  if (error) return <ErrorState error={error} onRetry={load} />;

  const riskById = Object.fromEntries((payload.risk.project_risks || []).map((item) => [item.project_id, item]));
  const states = [...new Set(payload.projects.map((project) => project.state).filter(Boolean))].sort();
  const filtered = payload.projects.filter((project) => (!stateFilter || project.state === stateFilter) && (!stageFilter || project.current_stage === stageFilter) && (!riskFilter || riskById[project.project_id]?.risk_band === riskFilter));
  const stageCounts = Object.fromEntries(["3a", "3A", "3D"].map((stage) => [stage, filtered.filter((project) => project.current_stage === stage).length]));
  const stageRisk = ["3a", "3A", "3D"].map((stage) => {
    const stageProjects = filtered.filter((project) => project.current_stage === stage);
    const average = stageProjects.length ? stageProjects.reduce((sum, project) => sum + (Number(riskById[project.project_id]?.risk_90d) || 0), 0) / stageProjects.length : 0;
    return { stage, projects: stageProjects.length, average };
  });
  const aging = [...filtered].filter((project) => Number(project.days_in_current_stage) >= 120).sort((a, b) => Number(b.days_in_current_stage) - Number(a.days_in_current_stage));
  const stateStage = states.map((state) => {
    const projects = filtered.filter((project) => project.state === state);
    return { state, total: projects.length, ...Object.fromEntries(["3a", "3A", "3D"].map((stage) => [stage, projects.filter((project) => project.current_stage === stage).length])) };
  }).filter((row) => row.total);

  return <div className="page sentinel-page">
    <section className="page-heading"><div><span className="section-label">NATIONAL STAGE SENTINEL</span><h2>Stage Progression &amp; Aging Review</h2><p>Observed latest stage duration and model-derived risk to support early workflow review. No statutory legal deadlines are inferred.</p></div><div className="data-status"><span>Filtered population</span><strong>{filtered.length} projects</strong></div></section>
    <section className="panel filters-panel"><PanelHeader eyebrow="FILTER SENTINEL VIEW" title="Narrow the monitoring population" /><div className="filters compact-filters"><label><span>State</span><select value={stateFilter} onChange={(event) => setStateFilter(event.target.value)}><option value="">All states</option>{states.map((state) => <option key={state}>{state}</option>)}</select></label><label><span>Stage</span><select value={stageFilter} onChange={(event) => setStageFilter(event.target.value)}><option value="">All stages</option>{["3a", "3A", "3D"].map((stage) => <option key={stage}>{stage}</option>)}</select></label><label><span>Risk</span><select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}><option value="">All risk bands</option>{["LOW", "MODERATE", "HIGH", "CRITICAL"].map((band) => <option key={band}>{band}</option>)}</select></label><button className="button secondary clear-filters" onClick={() => { setStateFilter(""); setStageFilter(""); setRiskFilter(""); }}>Clear filters</button></div></section>
    <section className="panel stage-national-panel"><PanelHeader eyebrow="LATEST OBSERVED STAGE" title="National Stage Distribution" detail="Counts update with the filters above." /><StageFlow counts={stageCounts} /></section>
    <section className="detail-two-column stage-analysis-grid"><article className="panel"><PanelHeader eyebrow="STAGE-WISE RISK" title="Average 90-Day Risk by Stage" /><div className="stage-risk-bars">{stageRisk.map((item) => <div key={item.stage}><div><strong>{item.stage}</strong><span>{item.projects} projects</span><b>{formatPercent(item.average)}</b></div><i><span style={{ width: `${Math.min(item.average, 100)}%` }} /></i></div>)}</div></article><article className="panel"><PanelHeader eyebrow="AGING THRESHOLD" title="Projects Stuck / Aging in Stage" detail="Latest observed duration of 120 days or more." /><div className="aging-summary"><strong>{aging.length}</strong><span>projects meet the prototype aging review threshold</span><p>Duration is an observed project-data indicator, not a statutory deadline.</p></div></article></section>
    <section className="panel"><PanelHeader eyebrow="LONGEST CURRENT STAGES" title="Priority Stage Review Queue" detail="Ordered by observed days in current stage." />{aging.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Project</th><th>State</th><th>Stage</th><th>Days in Stage</th><th>90-Day Risk</th><th>Band</th><th /></tr></thead><tbody>{aging.slice(0, 25).map((project) => { const risk = riskById[project.project_id] || {}; return <tr key={project.project_id}><td><strong>{project.project_name}</strong><small>{project.project_id}</small></td><td>{project.state}</td><td><span className="stage-pill">{project.current_stage}</span></td><td>{project.days_in_current_stage}</td><td>{formatPercent(risk.risk_90d)}</td><td><RiskBadge band={risk.risk_band} compact /></td><td><Link className="text-link" to={`/projects/${project.project_id}`}>Review</Link></td></tr>; })}</tbody></table></div> : <div className="empty-state">No project in this filtered population meets the 120-day prototype aging review threshold.</div>}</section>
    <section className="panel"><PanelHeader eyebrow="STATE-WISE STAGE DISTRIBUTION" title="State and Stage Matrix" /><div className="table-scroll"><table className="data-table"><thead><tr><th>State</th><th>Projects</th><th>3a</th><th>3A</th><th>3D</th></tr></thead><tbody>{stateStage.map((row) => <tr key={row.state}><td><strong>{row.state}</strong></td><td>{row.total}</td><td>{row["3a"]}</td><td>{row["3A"]}</td><td>{row["3D"]}</td></tr>)}</tbody></table></div></section>
    <MethodologyNote lastUpdated={payload.overview.last_updated} />
  </div>;
}

export default StageSentinelPage;
