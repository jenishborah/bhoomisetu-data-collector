import { useCallback, useEffect, useMemo, useState } from "react";

import ProjectTable from "../components/ProjectTable";
import { ErrorState, LoadingState, MethodologyNote, PanelHeader } from "../components/Ui";
import { getProjects, getRiskOverview } from "../services/api";

const PAGE_SIZE = 20;

function Projects() {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const [search, setSearch] = useState("");
  const [stateFilter, setStateFilter] = useState("");
  const [stageFilter, setStageFilter] = useState("");
  const [riskFilter, setRiskFilter] = useState("");
  const [page, setPage] = useState(1);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [projectResponse, riskResponse] = await Promise.all([getProjects({ limit: 1000 }), getRiskOverview()]);
      setPayload({ projects: projectResponse.projects || [], risk: riskResponse });
    } catch (err) { setError(err.message); }
  }, []);
  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);
  const riskById = useMemo(() => Object.fromEntries((payload?.risk?.project_risks || []).map((item) => [item.project_id, item])), [payload]);
  const stateOptions = useMemo(() => [...new Set((payload?.projects || []).map((item) => item.state).filter(Boolean))].sort(), [payload]);
  const filtered = useMemo(() => {
    const needle = search.trim().toLowerCase();
    return (payload?.projects || []).filter((project) => {
      const risk = riskById[project.project_id] || {};
      const searchable = [project.project_id, project.project_name, project.state, project.district, project.implementing_agency].join(" ").toLowerCase();
      return (!needle || searchable.includes(needle))
        && (!stateFilter || project.state === stateFilter)
        && (!stageFilter || project.current_stage === stageFilter)
        && (!riskFilter || risk.risk_band === riskFilter);
    }).sort((a, b) => (Number(riskById[b.project_id]?.risk_90d) || 0) - (Number(riskById[a.project_id]?.risk_90d) || 0));
  }, [payload, riskById, search, stateFilter, stageFilter, riskFilter]);
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const visible = filtered.slice((safePage - 1) * PAGE_SIZE, safePage * PAGE_SIZE);
  const changeFilter = (setter) => (event) => { setter(event.target.value); setPage(1); };

  if (!payload && !error) return <LoadingState label="Loading the national project register…" />;
  if (error) return <ErrorState error={error} onRetry={load} />;

  return <div className="page projects-page">
    <section className="page-heading"><div><span className="section-label">NATIONAL PROJECT REGISTER</span><h2>Projects</h2><p>Searchable latest-state register. Each row represents one project, not every historical snapshot.</p></div><div className="data-status"><span>Records loaded</span><strong>{payload.projects.length.toLocaleString()} projects</strong></div></section>
    <section className="panel filters-panel">
      <PanelHeader eyebrow="FILTER PROJECT REGISTER" title="Find a project" detail="Filter locally loaded project and calibrated-risk data." />
      <div className="filters">
        <label className="search-field"><span>Search project, state, district or agency</span><input value={search} onChange={changeFilter(setSearch)} placeholder="e.g. SYN-00001 or Assam" /></label>
        <label><span>State</span><select value={stateFilter} onChange={changeFilter(setStateFilter)}><option value="">All states</option>{stateOptions.map((state) => <option key={state}>{state}</option>)}</select></label>
        <label><span>Stage</span><select value={stageFilter} onChange={changeFilter(setStageFilter)}><option value="">All stages</option>{["3a", "3A", "3D"].map((stage) => <option key={stage}>{stage}</option>)}</select></label>
        <label><span>90-day risk band</span><select value={riskFilter} onChange={changeFilter(setRiskFilter)}><option value="">All bands</option>{["LOW", "MODERATE", "HIGH", "CRITICAL"].map((band) => <option key={band}>{band}</option>)}</select></label>
        <button className="button secondary clear-filters" onClick={() => { setSearch(""); setStateFilter(""); setStageFilter(""); setRiskFilter(""); setPage(1); }}>Clear filters</button>
      </div>
    </section>
    <section className="panel project-register-panel">
      <PanelHeader eyebrow="LATEST PROJECT STATE" title={`${filtered.length.toLocaleString()} matching project${filtered.length === 1 ? "" : "s"}`} aside={<span className="panel-count">Page {safePage} of {totalPages}</span>} />
      <ProjectTable projects={visible} riskById={riskById} showPagination />
      {filtered.length > 0 && <div className="pagination"><button className="button secondary" disabled={safePage <= 1} onClick={() => setPage((current) => Math.max(1, current - 1))}>Previous</button><span>Showing {(safePage - 1) * PAGE_SIZE + 1}–{Math.min(safePage * PAGE_SIZE, filtered.length)} of {filtered.length}</span><button className="button secondary" disabled={safePage >= totalPages} onClick={() => setPage((current) => Math.min(totalPages, current + 1))}>Next</button></div>}
    </section>
    <MethodologyNote lastUpdated={payload.risk.last_updated} />
  </div>;
}

export default Projects;
