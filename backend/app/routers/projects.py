"""Project, risk, evidence, action and simulation endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from backend.app.repositories.project_repository import get_repository
from backend.app.schemas import SimulationRequest
from backend.app.services.action_engine import get_recommended_actions
from backend.app.services.evidence_engine import get_project_evidence
from backend.app.services.predictor import PredictionError, predict_project
from backend.app.services.risk_engine import build_project_risk_response
from backend.app.services.simulator import simulate_project


router = APIRouter(prefix="/projects", tags=["Projects"])


def _latest_snapshot_or_404(project_id: str) -> dict:
    snapshot = get_repository().get_latest_snapshot(project_id)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    return snapshot


@router.get("", summary="List latest state for projects")
def list_projects(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=200),
    state: str | None = Query(default=None, max_length=100),
    stage: str | None = Query(default=None, max_length=20),
):
    repository = get_repository()
    return {
        "count": repository.count_projects(search=search, state=state, stage=stage),
        "limit": limit,
        "offset": offset,
        "projects": repository.list_projects(limit=limit, offset=offset, search=search, state=state, stage=stage),
    }


@router.get("/{project_id}/snapshots", summary="Get temporal snapshots for a project")
def get_project_snapshots(project_id: str):
    snapshots = get_repository().get_project_snapshots(project_id)
    if not snapshots:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    return {"project_id": project_id, "count": len(snapshots), "snapshots": snapshots}


@router.post("/{project_id}/predict", summary="Predict calibrated 30, 60 and 90 day delay probabilities")
def predict_project_risk(project_id: str):
    snapshot = _latest_snapshot_or_404(project_id)
    try:
        prediction = predict_project(snapshot)
    except (PredictionError, ValueError):
        raise HTTPException(status_code=500, detail="The prototype model could not score this project.") from None
    return {
        "project_id": project_id,
        "snapshot_id": snapshot.get("snapshot_id"),
        "current_stage": snapshot.get("current_stage"),
        **prediction,
    }


@router.get("/{project_id}/risk", summary="Get risk bands and trend for a project")
def get_project_risk(project_id: str):
    snapshot = _latest_snapshot_or_404(project_id)
    try:
        prediction = predict_project(snapshot)
        risk = build_project_risk_response(prediction)
    except (PredictionError, ValueError):
        raise HTTPException(status_code=500, detail="The prototype model could not calculate project risk.") from None
    return {
        "project_id": project_id,
        "snapshot_id": snapshot.get("snapshot_id"),
        "current_stage": snapshot.get("current_stage"),
        **risk,
    }


@router.get("/{project_id}/evidence", summary="Get saved local SHAP explanation for a project")
def get_project_evidence_route(project_id: str):
    snapshot = _latest_snapshot_or_404(project_id)
    try:
        return get_project_evidence(snapshot)
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Saved SHAP explanation artifacts are unavailable.") from None


@router.get("/{project_id}/actions", summary="Get deterministic recommended actions")
def get_project_actions(project_id: str):
    snapshot = _latest_snapshot_or_404(project_id)
    return {
        "project_id": project_id,
        "snapshot_id": snapshot.get("snapshot_id"),
        "recommendations": get_recommended_actions(snapshot),
        "note": "Recommendations are deterministic prototype workflow prompts and do not guarantee a risk reduction or project outcome.",
    }


@router.post("/{project_id}/simulate", summary="Run a model-based intervention scenario")
def simulate_project_route(project_id: str, payload: SimulationRequest):
    snapshot = _latest_snapshot_or_404(project_id)
    changes = payload.model_dump(exclude_none=True)
    if not changes:
        raise HTTPException(status_code=422, detail="Provide at least one editable scenario field.")
    try:
        return simulate_project(snapshot, changes)
    except (PredictionError, ValueError):
        raise HTTPException(status_code=500, detail="The prototype model could not simulate this scenario.") from None


@router.get("/{project_id}", summary="Get latest enriched project record")
def get_project(project_id: str):
    project = get_repository().get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found.")
    return {"project": project}
