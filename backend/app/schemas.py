from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Project
# ============================================================

class ProjectSummary(BaseModel):
    project_id: str
    project_name: Optional[str] = None
    state: Optional[str] = None
    district: Optional[str] = None
    current_stage: Optional[str] = None

    land_required_ha: Optional[float] = None
    land_to_acquire_ha: Optional[float] = None
    total_parcels: Optional[int] = None
    affected_families: Optional[int] = None


class ProjectDetail(ProjectSummary):
    snapshot_id: Optional[str] = None
    snapshot_date: Optional[str] = None

    vulnerable_families: Optional[int] = None

    pending_approvals_count: Optional[int] = None
    file_pending_days: Optional[float] = None
    docs_complete_pct: Optional[float] = None

    court_cases_count: Optional[int] = None
    case_age_months: Optional[float] = None
    title_disputes_parcels: Optional[int] = None

    grievances_30d: Optional[int] = None
    grievance_redressal_days: Optional[float] = None

    compensation_awarded_cr: Optional[float] = None
    compensation_disbursed_cr: Optional[float] = None
    disbursal_pct: Optional[float] = None

    rnr_progress_pct: Optional[float] = None
    rnr_consent_pct: Optional[float] = None

    forest_involved: Optional[int] = None
    proximity_km_to_urban: Optional[float] = None

    approval_dept: Optional[str] = None
    terrain_type: Optional[str] = None
    urban_rural: Optional[str] = None


# ============================================================
# Prediction
# ============================================================

class HorizonRisk(BaseModel):
    raw_probability: float
    calibrated_probability: float


class PredictionResponse(BaseModel):
    project_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    current_stage: Optional[str] = None

    predictions: dict[str, HorizonRisk]

    model_version: str


# ============================================================
# Risk
# ============================================================

class RiskHorizon(BaseModel):
    probability: float
    percentage: float
    score: int
    band: str


class RiskThresholds(BaseModel):
    low: str
    moderate: str
    high: str
    critical: str
    note: str


class RiskSummary(BaseModel):
    headline_band: str
    headline_score: int
    headline_percentage: float

    trend: str

    horizons: dict[str, RiskHorizon]

    thresholds: RiskThresholds


class ProjectRiskResponse(BaseModel):
    project_id: Optional[str] = None
    snapshot_id: Optional[str] = None
    current_stage: Optional[str] = None

    risk: RiskSummary

    model_version: Optional[str] = None


# ============================================================
# Generic API response
# ============================================================

class APIStatus(BaseModel):
    status: str
    message: Optional[str] = None
    data: Optional[Any] = None


# ============================================================
# What-if simulation
# ============================================================

class SimulationRequest(BaseModel):
    """Safe, editable operational fields for a model-based scenario."""

    model_config = ConfigDict(extra="forbid")

    pending_approvals_count: Optional[int] = Field(default=None, ge=0, le=100)
    file_pending_days: Optional[float] = Field(default=None, ge=0, le=3650)
    docs_complete_pct: Optional[float] = Field(default=None, ge=0, le=100)
    disbursal_pct: Optional[float] = Field(default=None, ge=0, le=100)
    rnr_progress_pct: Optional[float] = Field(default=None, ge=0, le=100)
    grievances_30d: Optional[int] = Field(default=None, ge=0, le=10000)
    grievance_redressal_days: Optional[float] = Field(default=None, ge=0, le=3650)
