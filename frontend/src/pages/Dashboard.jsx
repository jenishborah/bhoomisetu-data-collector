import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";

import StageFlow from "../components/StageFlow";
import { ErrorState, KpiCard, LoadingState, MethodologyNote, PanelHeader, RiskBadge } from "../components/Ui";
import { formatPercent } from "../utils/format";
import { getDashboardOverview, getRiskOverview } from "../services/api";

function Dashboard() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [overview, risk] = await Promise.all([getDashboardOverview(), getRiskOverview()]);
      setData({ overview, risk });
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);
  if (!data && !error) return <LoadingState label="Loading national monitoring data…" />;
  if (error) return <ErrorState error={error} onRetry={load} />;

  const { overview, risk } = data;
  const summary = overview.risk_summary || {};
  const distribution = risk.risk_distribution || {};
  const stateRows = overview.projects_by_state || [];
  const highest = Math.max(...stateRows.map((row) => row.ongoing_projects), 1);
  const priorityProjects = risk.top_high_priority_projects || risk.high_priority_projects || [];
  const stateRisk = risk.state_risk_distribution || risk.state_risk || [];

  return <div className="page dashboard-page">
    <section className="page-heading">
      <div><span className="section-label">NATIONAL MONITORING COMMAND CENTER</span><h2>Land Acquisition Project Dashboard</h2><p>Current project state, model-derived early-warning risk, and workflow priorities from local prototype datasets.</p></div>
      <div className="data-status"><span>Latest prototype data snapshot</span><strong>{overview.last_updated || "Available"}</strong></div>
    </section>

    <section className="kpi-grid" aria-label="National key performance indicators">
      <KpiCard label="Total Projects" value={overview.total_projects?.toLocaleString?.() ?? overview.total_projects} note="Unique projects in local master data" />
      <KpiCard label="Ongoing Projects" value={overview.ongoing_projects?.toLocaleString?.() ?? overview.ongoing_projects} note="Latest snapshot has an active stage" tone="green" />
      <KpiCard label="High Risk" value={summary.high ?? 0} note="90-day prototype risk band" tone="saffron" />
      <KpiCard label="Critical" value={summary.critical ?? 0} note="Requires early review" tone="red" />
    </section>

    <section className="dashboard-grid primary-grid">
      <article className="panel state-panel">
        <PanelHeader eyebrow="GEOGRAPHIC DISTRIBUTION" title="Ongoing Projects by State" aside={<span className="panel-count">{stateRows.length} States</span>} />
        <div className="state-bars">
          {stateRows.map((row, index) => <div className="state-bar-row" key={row.state}>
            <span className="rank">{String(index + 1).padStart(2, "0")}</span><strong>{row.state}</strong>
            <div className="bar-track" aria-label={`${row.state}: ${row.ongoing_projects} ongoing projects`}><span className="bar-fill" style={{ width: `${(row.ongoing_projects / highest) * 100}%` }} /></div>
            <span className="bar-value">{row.ongoing_projects}</span>
          </div>)}
        </div>
      </article>
      <article className="panel stage-panel">
        <PanelHeader eyebrow="ACQUISITION PIPELINE" title="Stage Sentinel" detail="Latest observed project stage; not a statutory deadline tracker." />
        <StageFlow counts={overview.projects_by_stage} />
        <div className="stage-footer"><span>3a → 3A → 3D</span><Link to="/stage-sentinel">Open national stage review</Link></div>
      </article>
    </section>

    <section className="panel risk-distribution-panel">
      <PanelHeader eyebrow="MODEL-DERIVED RISK" title="National Risk Overview" detail="Headline bands use calibrated 90-day prototype probabilities." />
      <div className="risk-distribution">
        {["LOW", "MODERATE", "HIGH", "CRITICAL"].map((band) => {
          const count = distribution[band] ?? 0;
          const width = overview.total_projects ? (count / overview.total_projects) * 100 : 0;
          return <div className={`risk-summary-item ${band.toLowerCase()}`} key={band}><div><RiskBadge band={band} /><strong>{count}</strong></div><span className="distribution-track"><i style={{ width: `${width}%` }} /></span><small>{formatPercent(width)} of scored projects</small></div>;
        })}
      </div>
    </section>

    <section className="panel priority-panel">
      <PanelHeader eyebrow="EARLY ATTENTION QUEUE" title="High Priority Projects" detail="Projects in HIGH or CRITICAL prototype bands, ordered by 90-day estimate." aside={<Link className="text-link" to="/projects">View all projects</Link>} />
      {priorityProjects.length ? <div className="table-scroll"><table className="data-table"><thead><tr><th>Project</th><th>State</th><th>Stage</th><th>90-Day Risk</th><th>Band</th><th>Trend</th><th /></tr></thead><tbody>{priorityProjects.slice(0, 10).map((item) => <tr key={item.project_id}><td><strong>{item.project_name || item.project_id}</strong><small>{item.project_id}</small></td><td>{item.state}<small>{item.district}</small></td><td><span className="stage-pill">{item.current_stage}</span></td><td>{formatPercent(item.risk_90d ?? item.risk_percentage)}</td><td><RiskBadge band={item.risk_band} compact /></td><td><span className={`trend ${String(item.trend).toLowerCase()}`}>{item.trend}</span></td><td><Link className="text-link" to={`/projects/${item.project_id}`}>Review</Link></td></tr>)}</tbody></table></div> : <div className="empty-state">No projects currently meet the HIGH or CRITICAL prototype threshold. The national score distribution remains available above.</div>}
    </section>

    <section className="panel state-risk-panel">
      <PanelHeader eyebrow="STATE RISK OVERVIEW" title="State-level Project Risk" detail="Counts use the latest snapshot once per project." />
      <div className="table-scroll"><table className="data-table"><thead><tr><th>State</th><th>Projects</th><th>High Risk</th><th>Critical</th><th>Average 90-Day Risk</th></tr></thead><tbody>{stateRisk.map((item) => <tr key={item.state}><td><strong>{item.state}</strong></td><td>{item.projects}</td><td>{item.high_risk_projects ?? item.high_risk ?? 0}</td><td>{item.critical_projects ?? item.critical ?? 0}</td><td><span className="average-risk"><i style={{ width: `${Math.min(Number(item.average_90d_risk) || 0, 100)}%` }} />{formatPercent(item.average_90d_risk)}</span></td></tr>)}</tbody></table></div>
    </section>

    <MethodologyNote lastUpdated={overview.last_updated} source={overview.data_source} />
  </div>;
}

export default Dashboard;
