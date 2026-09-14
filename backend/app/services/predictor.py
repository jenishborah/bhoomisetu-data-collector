from __future__ import annotations

import os

# Render free-tier instances have limited CPU/RAM.
# Keep numerical/ML inference single-threaded to avoid excessive
# joblib/OpenMP worker creation during dashboard batch prediction.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


# ================================================================
# BhoomiSetu Predictor Service
# ================================================================
# Location expected:
#   backend/app/services/predictor.py
#
# Project structure expected:
#   output/
#     synthetic/
#       ml/
#         baseline/models/
#         calibration/models/
#
# This service DOES NOT train models.
# It loads the already-trained XGBoost pipelines and applies the
# already-fitted sigmoid calibration layer.
# ================================================================


PROJECT_ROOT = Path(__file__).resolve().parents[3]

ML_DIR = PROJECT_ROOT / "output" / "synthetic" / "ml"

MODEL_DIR = ML_DIR / "baseline" / "models"
CALIBRATION_DIR = ML_DIR / "calibration" / "models"


TARGETS = {
    "30d": "delay_next_30d",
    "60d": "delay_next_60d",
    "90d": "delay_next_90d",
}


# Exact raw predictor set used by the existing ML feature builder.
PREDICTOR_COLUMNS = [
    "days_in_current_stage",
    "days_since_project_start",
    "land_required_ha",
    "land_to_acquire_ha",
    "total_parcels",
    "affected_families",
    "vulnerable_families",
    "pending_approvals_count",
    "file_pending_days",
    "docs_complete_pct",
    "court_cases_count",
    "case_age_months",
    "title_disputes_parcels",
    "grievances_30d",
    "compensation_awarded_cr",
    "compensation_disbursed_cr",
    "disbursal_pct",
    "rnr_progress_pct",
    "rnr_consent_pct",
    "grievance_redressal_days",
    "forest_involved",
    "proximity_km_to_urban",
    "land_acquisition_ratio",
    "affected_family_per_parcel",
    "vulnerable_family_share",
    "compensation_pending_cr",
    "rnr_remaining_pct",
    "docs_remaining_pct",
    "current_stage",
    "approval_dept",
    "terrain_type",
    "urban_rural",
]


class ModelLoadError(RuntimeError):
    """Raised when a required ML artifact cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when a project snapshot cannot be scored."""


class Predictor:
    """
    Loads BhoomiSetu's existing XGBoost models and sigmoid
    calibration artifacts.

    The saved sklearn pipeline is responsible for preprocessing
    categorical and numeric variables exactly as during training.
    """

    def __init__(self) -> None:
        self.models: dict[str, Any] = {}
        self.calibrators: dict[str, dict[str, float]] = {}

        for horizon, target in TARGETS.items():
            self.models[horizon] = self._load_model(target)
            self.calibrators[horizon] = self._load_calibrator(target)

    # ------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------

    @staticmethod
    def _load_model(target: str) -> Any:
        path = MODEL_DIR / f"{target}_xgboost.joblib"

        if not path.exists():
            raise ModelLoadError(
                f"XGBoost model not found for {target}: {path}"
            )

        try:
            model = joblib.load(path)

            # Render free-tier instances have limited CPU/RAM.
            # Force every exposed sklearn/XGBoost n_jobs parameter
            # to one worker during API inference.
            try:
                params = model.get_params(deep=True)
                single_thread_params = {
                    name: 1
                    for name in params
                    if name == "n_jobs" or name.endswith("__n_jobs")
                }

                if single_thread_params:
                    model.set_params(**single_thread_params)
            except Exception:
                # Some saved estimators/pipelines may not expose
                # set_params(). Do not fail artifact loading for that.
                pass

            return model
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load XGBoost model for {target}: {path}\n"
                f"Reason: {exc}"
            ) from exc

    @staticmethod
    def _load_calibrator(target: str) -> dict[str, float]:
        path = CALIBRATION_DIR / f"{target}_sigmoid.joblib"

        if not path.exists():
            raise ModelLoadError(
                f"Sigmoid calibration artifact not found for {target}: {path}"
            )

        try:
            artifact = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load calibration artifact for {target}: {path}\n"
                f"Reason: {exc}"
            ) from exc

        # Current portable artifact format.
        if isinstance(artifact, dict):
            required = [
                "coefficient",
                "intercept",
                "epsilon",
            ]

            missing = [
                key for key in required
                if key not in artifact
            ]

            if missing:
                raise ModelLoadError(
                    f"Calibration artifact {path} is missing: {missing}"
                )

            return {
                "coefficient": float(artifact["coefficient"]),
                "intercept": float(artifact["intercept"]),
                "epsilon": float(artifact["epsilon"]),
            }

        # Backward compatibility for the older custom
        # SigmoidCalibrator class.
        if (
            hasattr(artifact, "model")
            and hasattr(artifact, "epsilon")
        ):
            return {
                "coefficient": float(
                    np.asarray(
                        artifact.model.coef_
                    ).reshape(-1)[0]
                ),
                "intercept": float(
                    np.asarray(
                        artifact.model.intercept_
                    ).reshape(-1)[0]
                ),
                "epsilon": float(
                    artifact.epsilon
                ),
            }

        raise ModelLoadError(
            f"Unsupported calibration artifact format: {path}"
        )

    # ------------------------------------------------------------
    # Feature preparation
    # ------------------------------------------------------------

    @staticmethod
    def _derive_fields(record: dict[str, Any]) -> dict[str, Any]:
        """
        Calculate engineered fields when they are not explicitly
        supplied by the API caller.

        The formulas mirror the existing feature-builder logic.
        """
        r = dict(record)

        # land_acquisition_ratio
        if "land_acquisition_ratio" not in r:
            if (
                "land_required_ha" in r
                and "land_to_acquire_ha" in r
            ):
                denominator = float(
                    r["land_required_ha"]
                )

                r["land_acquisition_ratio"] = (
                    float(r["land_to_acquire_ha"]) / denominator
                    if denominator > 0
                    else 0.0
                )

        # affected_family_per_parcel
        if "affected_family_per_parcel" not in r:
            if (
                "affected_families" in r
                and "total_parcels" in r
            ):
                denominator = float(
                    r["total_parcels"]
                )

                r["affected_family_per_parcel"] = (
                    float(r["affected_families"]) / denominator
                    if denominator > 0
                    else 0.0
                )

        # vulnerable_family_share
        if "vulnerable_family_share" not in r:
            if (
                "vulnerable_families" in r
                and "affected_families" in r
            ):
                denominator = float(
                    r["affected_families"]
                )

                r["vulnerable_family_share"] = (
                    float(r["vulnerable_families"]) / denominator
                    if denominator > 0
                    else 0.0
                )

        # compensation_pending_cr
        if "compensation_pending_cr" not in r:
            if (
                "compensation_awarded_cr" in r
                and "compensation_disbursed_cr" in r
            ):
                r["compensation_pending_cr"] = max(
                    float(r["compensation_awarded_cr"])
                    - float(r["compensation_disbursed_cr"]),
                    0.0,
                )

        # rnr_remaining_pct
        if "rnr_remaining_pct" not in r:
            if "rnr_progress_pct" in r:
                r["rnr_remaining_pct"] = max(
                    100.0
                    - float(r["rnr_progress_pct"]),
                    0.0,
                )

        # docs_remaining_pct
        if "docs_remaining_pct" not in r:
            if "docs_complete_pct" in r:
                r["docs_remaining_pct"] = max(
                    100.0
                    - float(r["docs_complete_pct"]),
                    0.0,
                )

        return r

    def _make_frame(
        self,
        record: dict[str, Any],
    ) -> pd.DataFrame:
        r = self._derive_fields(record)

        missing = [
            column
            for column in PREDICTOR_COLUMNS
            if column not in r
        ]

        if missing:
            raise PredictionError(
                "Missing predictor fields:\n- "
                + "\n- ".join(missing)
            )

        row = {
            column: r[column]
            for column in PREDICTOR_COLUMNS
        }

        return pd.DataFrame([row])

    # ------------------------------------------------------------
    # Sigmoid calibration
    # ------------------------------------------------------------

    @staticmethod
    def _apply_sigmoid(
        raw_probability: float,
        calibrator: dict[str, float],
    ) -> float:
        epsilon = calibrator["epsilon"]
        coefficient = calibrator["coefficient"]
        intercept = calibrator["intercept"]

        p = float(
            np.clip(
                raw_probability,
                epsilon,
                1.0 - epsilon,
            )
        )

        logit = np.log(
            p / (1.0 - p)
        )

        z = (
            coefficient * logit
            + intercept
        )

        # Numerically stable sigmoid.
        if z >= 0:
            return float(
                1.0 / (1.0 + np.exp(-z))
            )

        exp_z = np.exp(z)

        return float(
            exp_z / (1.0 + exp_z)
        )

    # ------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------

    def predict(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        X = self._make_frame(record)

        predictions: dict[str, dict[str, float]] = {}

        for horizon, target in TARGETS.items():

            model = self.models[horizon]

            try:
                probability_matrix = (
                    model.predict_proba(X)
                )
            except Exception as exc:
                raise PredictionError(
                    f"Prediction failed for {target}: {exc}"
                ) from exc

            if probability_matrix.ndim != 2:
                raise PredictionError(
                    f"Unexpected probability output for {target}."
                )

            if probability_matrix.shape[1] < 2:
                raise PredictionError(
                    f"Model for {target} is not a binary classifier."
                )

            raw_probability = float(
                probability_matrix[0, 1]
            )

            calibrated_probability = (
                self._apply_sigmoid(
                    raw_probability,
                    self.calibrators[horizon],
                )
            )

            predictions[horizon] = {
                "raw_probability": raw_probability,
                "calibrated_probability": calibrated_probability,
            }

        return {
            "predictions": predictions,
            "model_version": (
                "synthetic-prototype-xgb-sigmoid-v1"
            ),
        }


    def predict_many(
        self,
        records: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Batch-score multiple project snapshots.

        The national dashboard uses this method so that each trained
        XGBoost pipeline processes all project rows in one call instead
        of calling predict_proba() once per project.
        """
        if not records:
            return []

        prepared_rows: list[dict[str, Any]] = []

        for record in records:
            r = self._derive_fields(record)

            missing = [
                column
                for column in PREDICTOR_COLUMNS
                if column not in r
            ]

            if missing:
                raise PredictionError(
                    "Missing predictor fields for "
                    f"{record.get('project_id', '<unknown>')}:\n- "
                    + "\n- ".join(missing)
                )

            prepared_rows.append(
                {
                    column: r[column]
                    for column in PREDICTOR_COLUMNS
                }
            )

        X = pd.DataFrame(prepared_rows)

        batch_probabilities: dict[str, np.ndarray] = {}

        for horizon, target in TARGETS.items():
            model = self.models[horizon]

            try:
                probability_matrix = model.predict_proba(X)
            except Exception as exc:
                raise PredictionError(
                    f"Batch prediction failed for {target}: {exc}"
                ) from exc

            if probability_matrix.ndim != 2:
                raise PredictionError(
                    f"Unexpected probability output for {target}."
                )

            if probability_matrix.shape[1] < 2:
                raise PredictionError(
                    f"Model for {target} is not a binary classifier."
                )

            if probability_matrix.shape[0] != len(records):
                raise PredictionError(
                    f"Model for {target} returned "
                    f"{probability_matrix.shape[0]} rows for "
                    f"{len(records)} records."
                )

            batch_probabilities[horizon] = probability_matrix[:, 1]

        results: list[dict[str, Any]] = []

        for index in range(len(records)):
            predictions: dict[str, dict[str, float]] = {}

            for horizon in TARGETS:
                raw_probability = float(
                    batch_probabilities[horizon][index]
                )

                calibrated_probability = self._apply_sigmoid(
                    raw_probability,
                    self.calibrators[horizon],
                )

                predictions[horizon] = {
                    "raw_probability": raw_probability,
                    "calibrated_probability": calibrated_probability,
                }

            results.append(
                {
                    "predictions": predictions,
                    "model_version": (
                        "synthetic-prototype-xgb-sigmoid-v1"
                    ),
                }
            )

        return results


# Lazy singleton.
# This prevents model loading during module import.
_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor

    if _predictor is None:
        _predictor = Predictor()

    return _predictor


def predict_project(
    record: dict[str, Any],
) -> dict[str, Any]:
    return get_predictor().predict(record)


def predict_projects(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Batch-score multiple project snapshots for dashboard aggregation."""
    return get_predictor().predict_many(records)
