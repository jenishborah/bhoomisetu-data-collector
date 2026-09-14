"""Model-based intervention scenario estimates using the existing predictor."""

from __future__ import annotations

from typing import Any

from backend.app.services.predictor import predict_project
from backend.app.services.risk_engine import build_risk_summary


def simulate_project(snapshot: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Re-score an edited copy of a latest snapshot without retraining a model."""
    current_prediction = predict_project(snapshot)
    current_risk = build_risk_summary(current_prediction["predictions"])

    scenario = dict(snapshot)
    scenario.update(changes)
    simulated_prediction = predict_project(scenario)
    simulated_risk = build_risk_summary(simulated_prediction["predictions"])

    comparisons: dict[str, dict[str, Any]] = {}
    for horizon in ("30d", "60d", "90d"):
        before = current_risk["horizons"][horizon]
        after = simulated_risk["horizons"][horizon]
        comparisons[horizon] = {
            "current_percentage": before["percentage"],
            "simulated_percentage": after["percentage"],
            "estimated_change_percentage_points": round(after["percentage"] - before["percentage"], 1),
            "risk_band_before": before["band"],
            "risk_band_after": after["band"],
        }

    return {
        "project_id": snapshot.get("project_id"),
        "snapshot_id": snapshot.get("snapshot_id"),
        "changed_fields": changes,
        "current": current_risk,
        "simulated": simulated_risk,
        "comparison": comparisons,
        "model_version": current_prediction["model_version"],
        "label": "Model-based scenario estimate",
        "note": "This simulation is not a causal guarantee.",
    }
