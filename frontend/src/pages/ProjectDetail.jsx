import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";

import StageFlow from "../components/StageFlow";
import { ErrorState, LoadingState, MethodologyNote, PanelHeader, RiskBadge } from "../components/Ui";
import { getActions, getEvidence, getProject, getProjectRisk, getProjectSnapshots, simulateProject } from "../services/api";
import { formatDate, formatPercent } from "../utils/format";

const simulatorFields = [
  ["pending_approvals_count", "Pending approvals", 0, 100, 1],
  ["file_pending_days", "File pending days", 0, 3650, 1],
  ["docs_complete_pct", "Documentation completeness (%)", 0, 100, 0.1],
  ["disbursal_pct", "Compensation disbursal (%)", 0, 100, 0.1],
  ["rnr_progress_pct", "R&R progress (%)", 0, 100, 0.1],
  ["grievances_30d", "Grievances in last 30 days", 0, 10000, 1],
  ["grievance_redressal_days", "Grievance resolution days", 0, 3650, 0.1],
];

function daysBetween(start, end) {
  const first = new Date(`${String(start).slice(0, 10)}T00:00:00`);
  const last = new Date(`${String(end).slice(0, 10)}T00:00:00`);
  return Number.isFinite(first.getTime()) && Number.isFinite(last.getTime()) ? Math.round((last - first) / 86400000) : null;
}

function ProjectDetail() {
  const { projectId } = useParams();
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const [form, setForm] = useState({});
  const [simulation, setSimulation] = useState(null);
  const [simulationError, setSimulationError] = useState(null);
  const [simulating, setSimulating] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    setSimulation(null);
    try {
      const [projectResponse, riskResponse, snapshotsResponse, evidenceResponse, actionsResponse] = await Promise.all([
        getProject(projectId), getProjectRisk(projectId), getProjectSnapshots(projectId), getEvidence(projectId), getActions(projectId),
      ]);
      const project = projectResponse.project;
      setPayload({ project, risk: riskResponse.risk, snapshots: snapshotsResponse.snapshots || [], evidence: evidenceResponse, actions: actionsResponse.recommendations || [] });
      setForm(Object.fromEntries(simulatorFields.map(([field]) => [field, project[field] ?? ""])));
    } catch (err) { setError(err.message); }
  }, [projectId]);
  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const updateForm = (field, value) => { setForm((current) => ({ ...current, [field]: value })); setSimulation(null); };
  const runSimulation = async (event) => {
    event.preventDefault();
    setSimulationError(null);
    setSimulating(true);
    try {
      const changes = Object.fromEntries(Object.entries(form).map(([field, value]) => [field, Number(value)]).filter(([, value]) => Number.isFinite(value)));
      const result = await simulateProject(projectId, changes);
      setSimulation(result);
    } catch (err) { setSimulationError(err.message); } finally { setSimulating(false); }
  };

  if (!payload && !error) return <LoadingState label={`Loading project ${projectId}…`} />;
  if (error) return <ErrorState error={error} onRetry={load} />;

  const { project, risk, snapshots, evidence, actions } = payload;
  const horizons = risk.horizons || {};
  const maxContribution = Math.max(...(evidence.drivers || []).map((driver) => driver.contribution_magnitude), 0.01);
  const firstSnapshot = snapshots[0];
  const observedDuration = firstSnapshot ? daysBetween(firstSnapshot.snapshot_date, project.snapshot_date) : null;

  return <div className="page detail-page">
    <p className="breadcrumb"><Link to="/projects">Projects</Link><span>›</span><span>{project.project_id}</span></p>
    <section className="project-header">
      <div><span className="section-label">PROJECT INTELLIGENCE RECORD</span><h2>{project.project_name || project.project_id}</h2><p className="project-id">{project.project_id}</p><div className="metadata-line"><span>{project.state}</span><span>{project.district}</span><span>{project.implementing_agency}</span><span>{project.project_type}</span></div></div>
      <div className="project-status"><span>Current Stage</span><strong>{project.current_stage}</strong><RiskBadge band={risk.headline_band} value={risk.headline_percentage} /><small>Last updated {formatDate(project.snapshot_date)}</small></div>
    </section>

    <section className="detail-overview-grid">
      <article className="panel overall-risk-panel"><PanelHeader eyebrow="MODEL-DERIVED RISK" title="Overall Risk" /><div className="overall-risk"><RiskBadge band={risk.headline_band} /><strong>{formatPercent(risk.headline_percentage)}</strong><span>90-day delay probability</span><div><small>Trend</small><b className={`trend ${String(risk.trend).toLowerCase()}`}>{risk.trend}</b></div></div></article>
      <article className="panel horizon-panel"><PanelHeader eyebrow="CALIBRATED HORIZONS" title="Delay Risk Outlook" /><div className="horizon-cards">{["30d", "60d", "90d"].map((horizon) => <div className="horizon-card" key={horizon}><span>{horizon.replace("d", " Days")}</span><strong>{formatPercent(horizons[horizon]?.percentage)}</strong><RiskBadge band={horizons[horizon]?.band} compact /><small>Prototype ML estimate</small></div>)}</div></article>
    </section>

    <section className="panel sentinel-detail-panel"><PanelHeader eyebrow="STAGE SENTINEL" title="Current stage monitoring" detail="Observed project duration is a prototype data observation, not a legal deadline." /><div className="stage-detail-layout"><StageFlow currentStage={project.current_stage} compact /><dl className="stage-facts"><div><dt>Current stage</dt><dd>{project.current_stage}</dd></div><div><dt>Days in current stage</dt><dd>{project.days_in_current_stage ?? "—"}</dd></div><div><dt>Stage entry date</dt><dd>{formatDate(project.stage_entry_date)}</dd></div><div><dt>Observed historical project duration</dt><dd>{observedDuration === null ? "Unavailable" : `${observedDuration} days`}</dd></div><div><dt>Stage status</dt><dd>Latest observed state</dd></div><div><dt>Next expected transition</dt><dd>Not inferred in this prototype</dd></div></dl></div></section>

    <section className="detail-two-column">
      <article className="panel evidence-panel"><PanelHeader eyebrow="EVIDENCE / WHY AT RISK" title="Why this project is flagged" detail="Model-derived contribution — not causal attribution." />
        {evidence.available ? <div className="evidence-list">{evidence.drivers.map((driver, index) => <div className="evidence-row" key={`${driver.feature}-${index}`}><div className="evidence-rank">{index + 1}</div><div className="evidence-main"><strong>{driver.label}</strong><small>{driver.category} · Current value: {String(driver.value ?? "—")}</small><div className="contribution-track"><i className={driver.direction} style={{ width: `${(driver.contribution_magnitude / maxContribution) * 100}%` }} /></div></div><div className="contribution-copy"><strong>{driver.shap_value > 0 ? "+" : ""}{driver.shap_value.toFixed(3)}</strong><small>model contribution</small></div></div>)}</div> : <div className="evidence-unavailable"><strong>Project-level SHAP evidence is unavailable for this snapshot.</strong><p>{evidence.unavailable_reason}</p><h3>Global model context</h3><ul>{(evidence.global_model_context || []).map((item) => <li key={item.feature}><span>{item.label}</span><strong>{formatPercent(item.normalized_importance_pct)}</strong></li>)}</ul></div>}
      </article>
      <article className="panel actions-panel"><PanelHeader eyebrow="RECOMMENDED ACTIONS" title="Next Best Actions" detail="Deterministic workflow prompts based on measurable latest conditions." />
        <div className="action-list">{actions.map((action, index) => <article className="action-row" key={`${action.action}-${index}`}><RiskBadge band={action.priority} compact /><div><strong>{action.action}</strong><p>{action.reason}</p><small><b>{action.owner}</b> · {action.category}</small></div><span className="action-status">{action.status}</span></article>)}</div>
      </article>
    </section>

    <section className="panel simulator-panel"><PanelHeader eyebrow="WHAT-IF SIMULATOR" title="Intervention Impact Simulator" detail="Model-based scenario estimate. This simulation is not a causal guarantee." />
      <form className="simulator-form" onSubmit={runSimulation}>{simulatorFields.map(([field, label, min, max, step]) => <label key={field}><span>{label}</span><input type="number" min={min} max={max} step={step} value={form[field] ?? ""} onChange={(event) => updateForm(field, event.target.value)} /></label>)}<div className="simulator-action"><button className="button primary" disabled={simulating}>{simulating ? "Calculating…" : "Run scenario estimate"}</button><small>Only the selected measurable fields are changed.</small></div></form>
      {simulationError && <p className="inline-error">{simulationError}</p>}
      {simulation && <div className="simulation-result"><div className="simulation-heading"><strong>Current vs simulated risk</strong><span>{simulation.label}</span></div><div className="simulation-comparison">{["30d", "60d", "90d"].map((horizon) => { const comparison = simulation.comparison[horizon]; return <div key={horizon}><span>{horizon.replace("d", " Days")}</span><p><strong>{formatPercent(comparison.current_percentage)}</strong><i>→</i><strong>{formatPercent(comparison.simulated_percentage)}</strong></p><small>{comparison.estimated_change_percentage_points > 0 ? "+" : ""}{comparison.estimated_change_percentage_points} percentage points</small><div><RiskBadge band={comparison.risk_band_before} compact /> <span>→</span> <RiskBadge band={comparison.risk_band_after} compact /></div></div>; })}</div><p>{simulation.note}</p></div>}
    </section>
    <MethodologyNote lastUpdated={project.snapshot_date} />
  </div>;
}

export default ProjectDetail;
