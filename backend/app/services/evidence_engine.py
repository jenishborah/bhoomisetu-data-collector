"""Deterministic local SHAP evidence for project-level explanations.

This service reads saved SHAP artifacts only. It never generates explanatory
text from an LLM and it never describes model associations as causal claims.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SHAP_DIR = PROJECT_ROOT / "output" / "synthetic" / "ml" / "shap"
LOCAL_DRIVERS_PATH = SHAP_DIR / "local_top_drivers.csv"
GLOBAL_IMPORTANCE_PATH = SHAP_DIR / "global_shap_importance.csv"
TARGET = "delay_next_90d"


FEATURE_INFO: dict[str, tuple[str, str]] = {
    "file_pending_days": ("Administrative file pending", "Administrative"),
    "docs_complete_pct": ("Documentation completeness", "Administrative"),
    "docs_remaining_pct": ("Documentation completeness", "Administrative"),
    "pending_approvals_count": ("Pending approvals", "Administrative"),
    "court_cases_count": ("Legal dispute burden", "Legal"),
    "case_age_months": ("Case age", "Legal"),
    "title_disputes_parcels": ("Title dispute parcels", "Legal"),
    "grievances_30d": ("Recent grievance load", "Community / Grievance"),
    "grievance_redressal_days": ("Grievance resolution time", "Community / Grievance"),
    "disbursal_pct": ("Compensation disbursal", "Compensation"),
    "compensation_pending_cr": ("Compensation pending", "Compensation"),
    "compensation_awarded_cr": ("Compensation awarded", "Compensation"),
    "compensation_disbursed_cr": ("Compensation disbursed", "Compensation"),
    "rnr_progress_pct": ("R&R implementation progress", "Rehabilitation & Resettlement"),
    "rnr_consent_pct": ("R&R consent progress", "Rehabilitation & Resettlement"),
    "rnr_remaining_pct": ("R&R implementation progress", "Rehabilitation & Resettlement"),
    "days_in_current_stage": ("Time in current stage", "Administrative"),
    "days_since_project_start": ("Project elapsed time", "Administrative"),
    "land_acquisition_ratio": ("Land acquisition progress", "Land Acquisition"),
    "land_required_ha": ("Land requirement", "Land Acquisition"),
    "land_to_acquire_ha": ("Land to acquire", "Land Acquisition"),
    "total_parcels": ("Affected land parcels", "Land Acquisition"),
    "affected_families": ("Affected families", "Community / Grievance"),
    "vulnerable_families": ("Vulnerable families", "Community / Grievance"),
}


def _feature_info(feature: str, display_feature: str | None = None) -> tuple[str, str, str | None]:
    if feature in FEATURE_INFO:
        label, category = FEATURE_INFO[feature]
        return label, category, feature
    if feature.startswith("current_stage_"):
        return f"Current stage: {feature.removeprefix('current_stage_')}", "Administrative", "current_stage"
    if feature.startswith("approval_dept_"):
        return "Approval department", "Administrative", "approval_dept"
    if feature.startswith("terrain_type_"):
        return "Terrain context", "Land Acquisition", "terrain_type"
    if feature.startswith("urban_rural_"):
        return "Settlement context", "Community / Grievance", "urban_rural"
    label = (display_feature or feature).replace("_", " ").strip().title()
    return label, "Administrative", feature


@lru_cache(maxsize=1)
def _local_drivers() -> pd.DataFrame:
    if not LOCAL_DRIVERS_PATH.exists():
        raise FileNotFoundError(f"Local SHAP driver file not found: {LOCAL_DRIVERS_PATH}")
    return pd.read_csv(LOCAL_DRIVERS_PATH)


@lru_cache(maxsize=1)
def _global_importance() -> pd.DataFrame:
    if not GLOBAL_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(f"Global SHAP importance file not found: {GLOBAL_IMPORTANCE_PATH}")
    return pd.read_csv(GLOBAL_IMPORTANCE_PATH)


def _public_driver(row: dict[str, Any], snapshot: dict[str, Any]) -> dict[str, Any]:
    feature = str(row["feature"])
    label, category, source_field = _feature_info(feature, row.get("display_feature"))
    raw_direction = str(row.get("direction") or "")
    direction = "increases_risk" if raw_direction == "risk_increasing" else "reduces_risk"
    shap_value = float(row.get("shap_value") or 0.0)
    value = snapshot.get(source_field) if source_field else row.get("feature_value")
    if value is None:
        value = row.get("feature_value")
    if direction == "increases_risk":
        explanation = f"{label} is contributing to the model's current risk score."
    else:
        explanation = f"{label} is reducing the model's current risk score."
    return {
        "feature": feature,
        "label": label,
        "category": category,
        "value": value,
        "shap_value": round(shap_value, 4),
        "contribution_magnitude": round(abs(shap_value), 4),
        "direction": direction,
        "explanation": explanation,
    }


def _global_context() -> list[dict[str, Any]]:
    frame = _global_importance()
    rows = frame[frame["target"] == TARGET].sort_values("rank").head(5).to_dict(orient="records")
    result = []
    for row in rows:
        label, category, _ = _feature_info(str(row["feature"]), row.get("display_feature"))
        result.append(
            {
                "feature": row["feature"],
                "label": label,
                "category": category,
                "normalized_importance_pct": round(float(row["normalized_importance_pct"]), 2),
            }
        )
    return result


def get_global_model_context() -> list[dict[str, Any]]:
    """Return saved global 90-day SHAP importance, clearly not local evidence."""
    return _global_context()


def get_project_evidence(snapshot: dict[str, Any], limit: int = 5) -> dict[str, Any]:
    """Return exact saved local drivers for the exact evaluation snapshot.

    Local SHAP output was generated only for held-out evaluation snapshots.
    Returning an unavailable state for the remainder is intentional: global
    importance is not substituted for a project-specific explanation.
    """
    snapshot_id = snapshot.get("snapshot_id")
    project_id = snapshot.get("project_id")
    rows = _local_drivers()
    matches = rows[(rows["target"] == TARGET) & (rows["snapshot_id"] == snapshot_id)].copy()
    matches = matches[matches["direction"] == "risk_increasing"]
    matches = matches.sort_values(["driver_rank", "shap_value"], ascending=[True, False]).head(limit)
    drivers = [_public_driver(row, snapshot) for row in matches.to_dict(orient="records")]
    available = bool(drivers)
    return {
        "project_id": project_id,
        "snapshot_id": snapshot_id,
        "horizon": "90d",
        "available": available,
        "drivers": drivers,
        "source": "Saved local SHAP output" if available else "No local SHAP output for this snapshot",
        "note": "Model-derived contribution — not causal attribution.",
        "unavailable_reason": None
        if available
        else (
            "A local SHAP explanation was not exported for this evaluation snapshot. "
            "The synthetic prototype only includes saved local explanations for a held-out subset; "
            "global feature importance is shown separately and is not a project-level explanation."
        ),
        "global_model_context": _global_context(),
    }
