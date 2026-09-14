"""National monitoring dashboard endpoints."""

from fastapi import APIRouter

from backend.app.services.dashboard_service import get_dashboard_service


router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("/overview", summary="National project monitoring overview")
def get_dashboard_overview():
    return get_dashboard_service().overview()


@router.get("/risk-overview", summary="National calibrated risk overview")
def get_dashboard_risk_overview():
    return get_dashboard_service().risk_overview()
