"""Efficient national dashboard aggregations built from latest project state."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from backend.app.repositories.project_repository import get_repository
from backend.app.services.evidence_engine import get_global_model_context
from backend.app.services.predictor import predict_project
from backend.app.services.risk_engine import build_risk_summary


MODEL_VERSION = "synthetic-prototype-xgb-sigmoid-v1"
PROTOTYPE_NOTE = (
    "Risk estimates are generated using the current prototype XGBoost models "
    "trained on synthetic temporal data. They are intended for demonstration "
    "and workflow validation, not operational decision-making."
)

import threading

class DashboardService:
    """Caches the expensive one-score-per-project national view for a process."""

    def __init__(self) -> None:
        self._scored_projects: list[dict[str, Any]] | None = None
        self._lock = threading.Lock()

    def _score_latest_projects(self) -> list[dict[str, Any]]:
        if self._scored_projects is not None:
            return self._scored_projects

        with self._lock:
            if self._scored_projects is not None:
                return self._scored_projects

            repository = get_repository()
            records = repository.latest_project_frame().to_dict(orient="records")
            scored: list[dict[str, Any]] = []
        for record in records:
            try:
                prediction = predict_project(record)
                risk = build_risk_summary(prediction["predictions"])
            except Exception:
                # Keep the national view usable if a future data refresh contains
                # one malformed record. Individual endpoints still report errors.
                continue

            score = dict(record)
            score["risk"] = risk
            score["risk_band"] = risk["headline_band"]
            score["risk_percentage"] = risk["headline_percentage"]
            score["risk_30d"] = risk["horizons"]["30d"]["percentage"]
            score["risk_60d"] = risk["horizons"]["60d"]["percentage"]
            score["risk_90d"] = risk["horizons"]["90d"]["percentage"]
            score["trend"] = risk["trend"]
            scored.append(score)

        self._scored_projects = scored
        return scored

    @staticmethod
    def _public_project(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "project_id": str(item["project_id"]),
            "project_name": item.get("project_name"),
            "state": item.get("state") or "Unknown",
            "district": item.get("district") or "Unknown",
            "implementing_agency": item.get("implementing_agency") or "Unknown",
            "project_type": item.get("project_type") or "Unknown",
            "current_stage": item.get("current_stage") or "Unknown",
            "days_in_current_stage": int(item.get("days_in_current_stage") or 0),
            "snapshot_date": str(item.get("snapshot_date"))[:10],
            "risk_band": item["risk_band"],
            "risk_percentage": item["risk_percentage"],
            "risk_30d": item["risk_30d"],
            "risk_60d": item["risk_60d"],
            "risk_90d": item["risk_90d"],
            "trend": item["trend"],
        }

    def _last_updated(self, projects: list[dict[str, Any]]) -> str | None:
        dates = [str(item.get("snapshot_date"))[:10] for item in projects if item.get("snapshot_date") is not None]
        return max(dates) if dates else None

    def overview(self) -> dict[str, Any]:
        projects = self._score_latest_projects()
        ongoing = [item for item in projects if item.get("current_stage")]
        bands = Counter(item["risk_band"].lower() for item in projects)
        stages = Counter(str(item.get("current_stage") or "Unknown") for item in ongoing)

        by_state: dict[str, dict[str, Any]] = defaultdict(lambda: {"ongoing_projects": 0, "high_risk_projects": 0, "critical_projects": 0, "risk_total": 0.0})
        by_agency: dict[str, int] = Counter()
        for item in ongoing:
            state = item.get("state") or "Unknown"
            entry = by_state[state]
            entry["ongoing_projects"] += 1
            entry["risk_total"] += float(item["risk_90d"])
            if item["risk_band"] == "HIGH":
                entry["high_risk_projects"] += 1
            elif item["risk_band"] == "CRITICAL":
                entry["critical_projects"] += 1
            by_agency[item.get("implementing_agency") or "Unknown"] += 1

        state_rows = [
            {
                "state": state,
                "ongoing_projects": values["ongoing_projects"],
                "high_risk_projects": values["high_risk_projects"],
                "critical_projects": values["critical_projects"],
                "average_90d_risk": round(values["risk_total"] / values["ongoing_projects"], 1),
            }
            for state, values in by_state.items()
        ]
        state_rows.sort(key=lambda row: (-row["ongoing_projects"], row["state"]))

        priority = [item for item in projects if item["risk_band"] in {"HIGH", "CRITICAL"}]
        priority.sort(key=lambda item: (str(item.get("snapshot_date")), item["risk_percentage"]), reverse=True)

        return {
            "total_projects": len(projects),
            "ongoing_projects": len(ongoing),
            "risk_summary": {key: bands.get(key, 0) for key in ("low", "moderate", "high", "critical")},
            "projects_by_state": state_rows,
            "projects_by_stage": {stage: stages.get(stage, 0) for stage in ("3a", "3A", "3D")},
            "projects_by_agency": [
                {"agency": agency, "ongoing_projects": count}
                for agency, count in sorted(by_agency.items(), key=lambda item: (-item[1], item[0]))
            ],
            "recently_flagged_projects": [self._public_project(item) for item in priority[:12]],
            "last_updated": self._last_updated(projects),
            "data_source": "Local synthetic_projects.csv and latest synthetic_snapshots.csv",
            "prototype_note": PROTOTYPE_NOTE,
        }

    def risk_overview(self) -> dict[str, Any]:
        projects = self._score_latest_projects()
        bands = Counter(item["risk_band"] for item in projects)
        groups: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"projects": 0, "low": 0, "moderate": 0, "high": 0, "critical": 0, "risk_total": 0.0}
        )
        for item in projects:
            state = item.get("state") or "Unknown"
            entry = groups[state]
            entry["projects"] += 1
            entry[item["risk_band"].lower()] += 1
            entry["risk_total"] += float(item["risk_90d"])

        state_rows = []
        for state, values in groups.items():
            state_rows.append(
                {
                    "state": state,
                    "projects": values["projects"],
                    "risk_summary": {key: values[key] for key in ("low", "moderate", "high", "critical")},
                    "high_risk_projects": values["high"],
                    "critical_projects": values["critical"],
                    "average_90d_risk": round(values["risk_total"] / values["projects"], 1),
                }
            )
        state_rows.sort(key=lambda row: (-row["critical_projects"], -row["high_risk_projects"], -row["average_90d_risk"], row["state"]))

        priority = [item for item in projects if item["risk_band"] in {"HIGH", "CRITICAL"}]
        priority.sort(key=lambda item: item["risk_percentage"], reverse=True)
        distribution = {key: bands.get(key, 0) for key in ("LOW", "MODERATE", "HIGH", "CRITICAL")}
        return {
            "projects_scored": len(projects),
            "risk_distribution": distribution,
            "state_risk_distribution": state_rows,
            "state_risk": state_rows,
            "project_risks": [self._public_project(item) for item in projects],
            "top_high_priority_projects": [self._public_project(item) for item in priority[:25]],
            "high_priority_projects": [self._public_project(item) for item in priority[:25]],
            "model_explanation_summary": get_global_model_context(),
            "model_version": MODEL_VERSION,
            "last_updated": self._last_updated(projects),
            "risk_basis": "Calibrated 30/60/90-day prototype model probabilities; headline band uses 90-day risk.",
            "prototype_note": PROTOTYPE_NOTE,
        }


_dashboard_service: DashboardService | None = None


def get_dashboard_service() -> DashboardService:
    global _dashboard_service
    if _dashboard_service is None:
        _dashboard_service = DashboardService()
    return _dashboard_service
