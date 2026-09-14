import { Link } from "react-router-dom";

import { EmptyState, RiskBadge } from "./Ui";
import { formatDate, formatPercent } from "../utils/format";

function ProjectTable({ projects, riskById = {}, showPagination = false }) {
  if (!projects?.length) return <EmptyState>No projects match the selected filters.</EmptyState>;
  return (
    <div className="table-scroll">
      <table className="data-table project-table">
        <thead><tr><th>Project ID</th><th>Project Name</th><th>State / District</th><th>Agency</th><th>Current Stage</th><th>90-Day Risk</th><th>Risk Band</th><th>Days in Stage</th><th>Last Updated</th></tr></thead>
        <tbody>
          {projects.map((project) => {
            const risk = riskById[project.project_id];
            const percentage = risk?.risk_90d ?? risk?.risk_percentage;
            const band = risk?.risk_band;
            return <tr key={project.project_id}>
              <td><Link className="table-link" to={`/projects/${project.project_id}`}>{project.project_id}</Link></td>
              <td><Link className="project-name" to={`/projects/${project.project_id}`}>{project.project_name || "Unnamed project"}</Link><small>{project.project_type || "Project type unavailable"}</small></td>
              <td>{project.state || "—"}<small>{project.district || "District unavailable"}</small></td>
              <td>{project.implementing_agency || "—"}</td>
              <td><span className="stage-pill">{project.current_stage || "—"}</span></td>
              <td>{percentage === undefined ? "—" : formatPercent(percentage)}</td>
              <td>{band ? <RiskBadge band={band} compact /> : <span className="muted">Loading</span>}</td>
              <td>{project.days_in_current_stage ?? "—"}</td>
              <td>{formatDate(project.snapshot_date)}</td>
            </tr>;
          })}
        </tbody>
      </table>
      {showPagination && <p className="table-hint">Select a project ID or name to open its detailed risk, evidence and scenario view.</p>}
    </div>
  );
}

export default ProjectTable;
