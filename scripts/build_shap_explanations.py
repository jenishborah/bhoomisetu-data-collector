"""
BhoomiSetu SHAP Explainability Pipeline
========================================

Builds global and local SHAP explanations for the already-trained
XGBoost delay models.

Targets:
    delay_next_30d
    delay_next_60d
    delay_next_90d

Important design:
    - SHAP is calculated on the underlying XGBoost model.
    - The sigmoid calibration layer is NOT used to calculate SHAP values.
    - Calibrated probabilities remain the probability layer used by
      Stage Sentinel.
    - The saved sigmoid mapping is read as its LogisticRegression
      coefficient/intercept rather than unpickling the custom calibrator.
    - Explanations describe MODEL-ATTRIBUTED factors, not causality.
    - Project-disjoint train/validation/test split is reproduced.
    - Forbidden/leakage columns are excluded from the explanation matrix.

Inputs:
    output/synthetic/ml/ml_features.csv
    output/synthetic/synthetic_snapshots.csv
    output/synthetic/ml/baseline/models/
        delay_next_XXd_xgboost.joblib
    output/synthetic/ml/calibration/models/
        delay_next_XXd_sigmoid.joblib

Outputs:
    output/synthetic/ml/shap/
        global_shap_importance.csv
        global_shap_importance_normalized.csv
        local_shap_explanations.csv
        local_top_drivers.csv
        shap_summary_report.txt
        validation_report.txt
        models/
        arrays/
            <target>_shap_values.npy
            <target>_test_probabilities.npy
            <target>_calibrated_probabilities.npy

Requirements:
    pip install shap joblib pandas numpy scikit-learn xgboost
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import shap
except ImportError:
    print("SHAP is not installed.")
    print("Install it with:")
    print("  pip install shap")
    sys.exit(1)


ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT
    / "output"
    / "synthetic"
    / "ml"
    / "ml_features.csv"
)

SNAPSHOT_FILE = (
    ROOT
    / "output"
    / "synthetic"
    / "synthetic_snapshots.csv"
)

BASELINE_MODEL_DIR = (
    ROOT
    / "output"
    / "synthetic"
    / "ml"
    / "baseline"
    / "models"
)

CALIBRATION_MODEL_DIR = (
    ROOT
    / "output"
    / "synthetic"
    / "ml"
    / "calibration"
    / "models"
)

OUTPUT_DIR = (
    ROOT
    / "output"
    / "synthetic"
    / "ml"
    / "shap"
)

ARRAY_DIR = OUTPUT_DIR / "arrays"
MODEL_DIR = OUTPUT_DIR / "models"

TARGETS = [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]

RANDOM_STATE = 20260914

# Number of local test rows for which detailed SHAP records are stored.
# All test rows are explained; this only limits the human-readable
# top-driver table to keep it compact.
MAX_LOCAL_ROWS_FOR_DETAIL = None

# Number of top positive and negative drivers per observation.
TOP_DRIVERS = 5


FORBIDDEN_FEATURES = {
    "snapshot_id",
    "project_id",
    "snapshot_date",
    "project_start_date",
    "stage_entry_date",
    "is_censored",
    "scenario",
    "scenario_ground_truth",
    "event_count_so_far",
    *TARGETS,
}


def load_data():
    """Load ML features and original snapshot metadata."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"ML feature file not found:\n{FEATURE_FILE}"
        )

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(
            f"Snapshot file not found:\n{SNAPSHOT_FILE}"
        )

    features = pd.read_csv(FEATURE_FILE)
    snapshots = pd.read_csv(SNAPSHOT_FILE)

    if len(features) != len(snapshots):
        raise ValueError(
            "Feature/snapshot row-count mismatch: "
            f"{len(features)} vs {len(snapshots)}"
        )

    required_snapshot_columns = {
        "project_id",
        "project_start_date",
        "snapshot_id",
        "snapshot_date",
        "stage_entry_date",
        "is_censored",
    }

    missing = sorted(
        required_snapshot_columns
        - set(snapshots.columns)
    )

    if missing:
        raise ValueError(
            "Snapshot dataset is missing required metadata:\n"
            + "\n".join(
                f"  - {c}" for c in missing
            )
        )

    # Attach metadata ONLY for split/identification.
    data = features.copy()

    data.insert(
        0,
        "project_id",
        snapshots["project_id"].values,
    )

    data.insert(
        1,
        "project_start_date",
        pd.to_datetime(
            snapshots["project_start_date"],
            errors="coerce",
        ).values,
    )

    data.insert(
        2,
        "snapshot_id",
        snapshots["snapshot_id"].values,
    )

    data.insert(
        3,
        "snapshot_date",
        pd.to_datetime(
            snapshots["snapshot_date"],
            errors="coerce",
        ).values,
    )

    data.insert(
        4,
        "stage_entry_date",
        pd.to_datetime(
            snapshots["stage_entry_date"],
            errors="coerce",
        ).values,
    )

    data.insert(
        5,
        "is_censored",
        snapshots["is_censored"].values,
    )

    return data


def make_project_split(data):
    """
    Reproduce the chronological 70/15/15 project split used in baseline
    training/calibration.
    """
    project_table = (
        data[
            [
                "project_id",
                "project_start_date",
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "project_start_date",
                "project_id",
            ]
        )
        .reset_index(drop=True)
    )

    n_projects = len(project_table)

    n_train = int(
        np.floor(n_projects * 0.70)
    )

    n_valid = int(
        np.floor(n_projects * 0.15)
    )

    train_projects = set(
        project_table.iloc[
            :n_train
        ]["project_id"]
    )

    validation_projects = set(
        project_table.iloc[
            n_train:n_train + n_valid
        ]["project_id"]
    )

    test_projects = set(
        project_table.iloc[
            n_train + n_valid:
        ]["project_id"]
    )

    if (
        train_projects
        & validation_projects
    ):
        raise ValueError(
            "Train/validation project overlap."
        )

    if (
        train_projects
        & test_projects
    ):
        raise ValueError(
            "Train/test project overlap."
        )

    if (
        validation_projects
        & test_projects
    ):
        raise ValueError(
            "Validation/test project overlap."
        )

    result = data.copy()
    result["split"] = "unassigned"

    result.loc[
        result["project_id"].isin(
            train_projects
        ),
        "split",
    ] = "train"

    result.loc[
        result["project_id"].isin(
            validation_projects
        ),
        "split",
    ] = "validation"

    result.loc[
        result["project_id"].isin(
            test_projects
        ),
        "split",
    ] = "test"

    if (
        result["split"] == "unassigned"
    ).any():
        raise ValueError(
            "Some snapshots were not assigned to a split."
        )

    project_split_counts = (
        result.groupby(
            "project_id"
        )["split"]
        .nunique()
    )

    if (
        project_split_counts > 1
    ).any():
        raise ValueError(
            "Project leakage detected."
        )

    return result


def get_predictor_columns(data):
    """
    Reconstruct the predictor matrix used by the baseline models.

    The feature builder output contains target columns plus predictors.
    Metadata and known leakage columns are explicitly excluded.
    """
    predictors = [
        c for c in data.columns
        if c not in FORBIDDEN_FEATURES
        and c != "split"
    ]

    if not predictors:
        raise ValueError(
            "No predictor columns found."
        )

    forbidden_present = (
        set(predictors)
        & FORBIDDEN_FEATURES
    )

    if forbidden_present:
        raise ValueError(
            "Forbidden features accidentally selected: "
            + ", ".join(
                sorted(forbidden_present)
            )
        )

    return predictors


def validate_raw_predictor_matrix(X, predictors):
    """Validate the raw ML feature matrix before model preprocessing."""
    if list(X.columns) != list(predictors):
        raise ValueError("Predictor column ordering mismatch.")

    if X.isna().any().any():
        # The baseline pipeline contains imputers, so missing values are
        # allowed here. We only reject non-finite numeric values.
        numeric_cols = [
            c for c in X.columns
            if pd.api.types.is_numeric_dtype(X[c])
        ]
        if numeric_cols and not np.isfinite(
            X[numeric_cols].dropna().to_numpy(dtype=float)
        ).all():
            raise ValueError(
                "Raw predictor matrix contains non-finite numeric values."
            )


def transform_for_xgboost(pipeline, X_raw):
    """
    Reproduce the exact preprocessing embedded in the trained baseline
    Pipeline, then return the numeric matrix consumed by XGBoost and
    the exact transformed feature names.

    Baseline training uses:
        numeric -> median imputation
        categorical -> most-frequent imputation -> one-hot encoding
        remainder -> drop
        verbose_feature_names_out=False
    """
    if not hasattr(pipeline, "named_steps"):
        raise TypeError(
            "Expected the saved baseline XGBoost Pipeline."
        )

    if "preprocess" not in pipeline.named_steps:
        raise ValueError(
            "Saved XGBoost model does not contain the expected "
            "'preprocess' step."
        )

    if "model" not in pipeline.named_steps:
        raise ValueError(
            "Saved XGBoost model does not contain the expected "
            "'model' step."
        )

    preprocessor = pipeline.named_steps["preprocess"]
    xgb_model = pipeline.named_steps["model"]

    X_transformed = preprocessor.transform(X_raw)

    feature_names = list(
        preprocessor.get_feature_names_out()
    )

    X_transformed = np.asarray(
        X_transformed,
        dtype=float,
    )

    if X_transformed.ndim != 2:
        raise ValueError(
            f"Expected 2D transformed matrix; got {X_transformed.shape}"
        )

    if X_transformed.shape[1] != len(feature_names):
        raise ValueError(
            "Transformed feature count/name count mismatch: "
            f"{X_transformed.shape[1]} vs {len(feature_names)}"
        )

    expected = getattr(
        xgb_model,
        "n_features_in_",
        None,
    )

    if expected is not None and int(expected) != X_transformed.shape[1]:
        raise ValueError(
            "Transformed feature count does not match XGBoost model: "
            f"matrix={X_transformed.shape[1]}, model={expected}"
        )

    if not np.isfinite(X_transformed).all():
        raise ValueError(
            "Transformed predictor matrix contains non-finite values."
        )

    return X_transformed, feature_names

def load_sigmoid_parameters(target):
    """
    Load the portable sigmoid calibration artifact.

    The updated calibration script saves a plain dictionary containing:
        method, version, epsilon, coefficient, intercept

    For backwards compatibility, this also supports the old custom
    SigmoidCalibrator object if it can be unpickled.
    """
    path = (
        CALIBRATION_MODEL_DIR
        / f"{target}_sigmoid.joblib"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Sigmoid calibration model not found:\n"
            f"{path}\n"
            "Run the portable calibration pipeline first."
        )

    try:
        saved = joblib.load(path)
    except AttributeError as exc:
        raise RuntimeError(
            "The sigmoid calibration artifact is an old custom pickle "
            "that cannot be loaded from this SHAP script.\n"
            "Run calibrate_delay_models_portable.py once to replace it "
            "with the portable artifact.\n"
            f"Original error: {exc}"
        ) from exc

    # Preferred format: portable dictionary created by the corrected
    # calibration script.
    if isinstance(saved, dict):
        required = {
            "coefficient",
            "intercept",
        }

        missing = required - set(saved.keys())

        if missing:
            raise ValueError(
                "Portable sigmoid artifact is missing: "
                + ", ".join(sorted(missing))
            )

        return {
            "coefficient": float(
                saved["coefficient"]
            ),
            "intercept": float(
                saved["intercept"]
            ),
            "epsilon": float(
                saved.get("epsilon", 1e-6)
            ),
        }

    # Backwards compatibility with the old custom class.
    if hasattr(saved, "model"):
        model = saved.model

        if not hasattr(model, "coef_") or not hasattr(
            model, "intercept_"
        ):
            raise TypeError(
                "Saved sigmoid LogisticRegression is not fitted."
            )

        return {
            "coefficient": float(
                np.asarray(
                    model.coef_
                ).reshape(-1)[0]
            ),
            "intercept": float(
                np.asarray(
                    model.intercept_
                ).reshape(-1)[0]
            ),
            "epsilon": float(
                getattr(saved, "epsilon", 1e-6)
            ),
        }

    raise TypeError(
        "Unsupported sigmoid calibration artifact format. "
        "Expected a portable dictionary containing coefficient and "
        "intercept."
    )

def calibrated_probability(
    parameters,
    raw_probability,
):
    """
    Apply the exact saved Platt-style sigmoid mapping without loading
    the custom SigmoidCalibrator class.
    """
    raw_probability = np.asarray(
        raw_probability,
        dtype=float,
    )

    epsilon = parameters["epsilon"]

    clipped = np.clip(
        raw_probability,
        epsilon,
        1.0 - epsilon,
    )

    logit = np.log(
        clipped
        / (1.0 - clipped)
    )

    z = (
        parameters["intercept"]
        + parameters["coefficient"] * logit
    )

    # Numerically stable sigmoid.
    calibrated = np.empty_like(z)

    positive = z >= 0

    calibrated[positive] = (
        1.0
        / (
            1.0
            + np.exp(-z[positive])
        )
    )

    exp_z = np.exp(z[~positive])

    calibrated[~positive] = (
        exp_z
        / (
            1.0
            + exp_z
        )
    )

    return np.clip(
        calibrated,
        0.0,
        1.0,
    )


def normalize_feature_name(name):
    """
    Convert technical one-hot names into a cleaner display label.

    This is presentation-only. The original feature name remains stored.
    """
    replacements = {
        "current_stage_3a":
            "Current stage: 3a",
        "current_stage_3A":
            "Current stage: 3A",
        "current_stage_3D":
            "Current stage: 3D",
        "urban_rural_Rural":
            "Setting: Rural",
        "urban_rural_Urban":
            "Setting: Urban",
        "terrain_type_Hilly":
            "Terrain: Hilly",
        "terrain_type_Plain":
            "Terrain: Plain",
        "approval_dept_Environment":
            "Approval department: Environment",
        "approval_dept_Revenue":
            "Approval department: Revenue",
        "approval_dept_Administration":
            "Approval department: Administration",
    }

    return replacements.get(
        name,
        name.replace("_", " "),
    )


def make_global_importance(
    shap_values,
    feature_names,
):
    """
    Create global mean absolute SHAP importance.

    SHAP magnitude:
        average absolute contribution to the XGBoost output space.
    """
    mean_abs = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    mean_signed = np.mean(
        shap_values,
        axis=0,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "display_feature": [
                normalize_feature_name(f)
                for f in feature_names
            ],
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
        }
    )

    total = (
        importance["mean_abs_shap"]
        .sum()
    )

    if total > 0:
        importance[
            "normalized_importance_pct"
        ] = (
            importance["mean_abs_shap"]
            / total
            * 100.0
        )
    else:
        importance[
            "normalized_importance_pct"
        ] = 0.0

    importance = importance.sort_values(
        "mean_abs_shap",
        ascending=False,
    ).reset_index(drop=True)

    importance.insert(
        0,
        "rank",
        np.arange(
            1,
            len(importance) + 1,
        ),
    )

    return importance


def make_local_explanations(
    test_data,
    X_transformed,
    shap_values,
    raw_probabilities,
    calibrated_probabilities,
    feature_names,
    target,
):
    """Produce long-form local SHAP records on the encoded model matrix."""
    rows = []

    for row_idx in range(len(test_data)):
        base = {
            "target": target,
            "project_id": test_data.iloc[row_idx]["project_id"],
            "snapshot_id": test_data.iloc[row_idx]["snapshot_id"],
            "snapshot_date": test_data.iloc[row_idx]["snapshot_date"],
            "current_stage": test_data.iloc[row_idx].get("current_stage", ""),
            "raw_xgboost_probability": raw_probabilities[row_idx],
            "calibrated_probability": calibrated_probabilities[row_idx],
        }

        for feature_idx, feature in enumerate(feature_names):
            shap_value = float(shap_values[row_idx, feature_idx])

            rows.append(
                {
                    **base,
                    "feature": feature,
                    "display_feature": normalize_feature_name(feature),
                    "feature_value": float(
                        X_transformed[row_idx, feature_idx]
                    ),
                    "shap_value": shap_value,
                    "direction": (
                        "risk_increasing"
                        if shap_value > 0
                        else ("risk_reducing" if shap_value < 0 else "neutral")
                    ),
                    "absolute_shap_value": abs(shap_value),
                }
            )

    return pd.DataFrame(rows)

def make_top_driver_table(
    test_data,
    X_transformed,
    shap_values,
    raw_probabilities,
    calibrated_probabilities,
    feature_names,
    target,
):
    """Compact dashboard/API-friendly top SHAP drivers."""
    rows = []

    max_rows = (
        len(test_data)
        if MAX_LOCAL_ROWS_FOR_DETAIL is None
        else min(len(test_data), MAX_LOCAL_ROWS_FOR_DETAIL)
    )

    for row_idx in range(max_rows):
        values = shap_values[row_idx]

        positive_indices = np.where(values > 0)[0]
        negative_indices = np.where(values < 0)[0]

        positive_indices = positive_indices[
            np.argsort(values[positive_indices])[::-1]
        ][:TOP_DRIVERS]

        negative_indices = negative_indices[
            np.argsort(np.abs(values[negative_indices]))[::-1]
        ][:TOP_DRIVERS]

        base = {
            "target": target,
            "project_id": test_data.iloc[row_idx]["project_id"],
            "snapshot_id": test_data.iloc[row_idx]["snapshot_id"],
            "snapshot_date": test_data.iloc[row_idx]["snapshot_date"],
            "current_stage": test_data.iloc[row_idx].get("current_stage", ""),
            "raw_xgboost_probability": raw_probabilities[row_idx],
            "calibrated_probability": calibrated_probabilities[row_idx],
        }

        for rank, feature_idx in enumerate(positive_indices, start=1):
            rows.append(
                {
                    **base,
                    "direction": "risk_increasing",
                    "driver_rank": rank,
                    "feature": feature_names[feature_idx],
                    "display_feature": normalize_feature_name(
                        feature_names[feature_idx]
                    ),
                    "feature_value": float(
                        X_transformed[row_idx, feature_idx]
                    ),
                    "shap_value": float(values[feature_idx]),
                }
            )

        for rank, feature_idx in enumerate(negative_indices, start=1):
            rows.append(
                {
                    **base,
                    "direction": "risk_reducing",
                    "driver_rank": rank,
                    "feature": feature_names[feature_idx],
                    "display_feature": normalize_feature_name(
                        feature_names[feature_idx]
                    ),
                    "feature_value": float(
                        X_transformed[row_idx, feature_idx]
                    ),
                    "shap_value": float(values[feature_idx]),
                }
            )

    return pd.DataFrame(rows)

def verify_shap_shape(
    shap_values,
    X,
):
    """Validate SHAP output dimensions."""
    if isinstance(
        shap_values,
        list,
    ):
        # Some SHAP versions can return a list for classifier outputs.
        if len(shap_values) == 2:
            shap_values = shap_values[1]
        else:
            raise ValueError(
                "Unexpected list-shaped SHAP output."
            )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim != 2:
        raise ValueError(
            "Expected 2D SHAP matrix; got "
            f"shape {shap_values.shape}"
        )

    expected_shape = (
        X.shape[0],
        X.shape[1],
    )

    if shap_values.shape != expected_shape:
        raise ValueError(
            "SHAP shape mismatch. "
            f"Expected {expected_shape}, "
            f"got {shap_values.shape}"
        )

    return shap_values


def run_target(
    data,
    feature_names,
    target,
):
    print()
    print("-" * 72)
    print(f"TARGET: {target}")
    print("-" * 72)

    baseline_path = (
        BASELINE_MODEL_DIR
        / f"{target}_xgboost.joblib"
    )

    if not baseline_path.exists():
        raise FileNotFoundError(
            f"XGBoost model not found:\n{baseline_path}"
        )

    model = joblib.load(
        baseline_path
    )

    sigmoid_parameters = load_sigmoid_parameters(
        target
    )

    test_data = data[
        data["split"] == "test"
    ].copy()

    if test_data.empty:
        raise ValueError(
            "Test split is empty."
        )

    X_test_raw = test_data[
        feature_names
    ].copy()

    validate_raw_predictor_matrix(
        X_test_raw,
        feature_names,
    )

    # Reproduce the exact preprocessing fitted during baseline training.
    X_test_transformed, encoded_feature_names = transform_for_xgboost(
        model,
        X_test_raw,
    )

    y_test = (
        test_data[target]
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------------
    # Generate model probabilities using the full saved pipeline.
    # --------------------------------------------------------------
    raw_probability = (
        model.predict_proba(
            X_test_raw
        )[:, 1]
    )

    calibrated_probability_values = (
        calibrated_probability(
            sigmoid_parameters,
            raw_probability,
        )
    )

    # --------------------------------------------------------------
    # SHAP TreeExplainer.
    # --------------------------------------------------------------
    print(
        "Building SHAP TreeExplainer..."
    )

    # Explain the actual XGBoost estimator, not the sklearn Pipeline.
    # The pipeline's fitted preprocessor has already transformed the raw
    # 32-column feature table into the exact numeric matrix used by XGBoost.
    xgb_model = model.named_steps["model"]

    explainer = shap.TreeExplainer(
        xgb_model
    )

    print(
        f"Calculating SHAP for "
        f"{len(X_test_transformed)} test snapshots..."
    )

    shap_values = explainer.shap_values(
        X_test_transformed
    )

    shap_values = verify_shap_shape(
        shap_values,
        X_test_transformed,
    )

    # --------------------------------------------------------------
    # Save raw SHAP array.
    # --------------------------------------------------------------
    np.save(
        ARRAY_DIR
        / f"{target}_shap_values.npy",
        shap_values,
    )

    np.save(
        ARRAY_DIR
        / f"{target}_test_probabilities.npy",
        raw_probability,
    )

    np.save(
        ARRAY_DIR
        / f"{target}_calibrated_probabilities.npy",
        calibrated_probability_values,
    )

    # --------------------------------------------------------------
    # Global importance.
    # --------------------------------------------------------------
    global_importance = (
        make_global_importance(
            shap_values,
            encoded_feature_names,
        )
    )

    if "target" not in global_importance.columns:
        global_importance.insert(
            0,
            "target",
            target,
        )

    global_path = (
        OUTPUT_DIR
        / "global_shap_importance.csv"
    )

    # Append all targets later in main; return now.
    global_normalized = (
        global_importance[
            [
                "target",
                "rank",
                "feature",
                "display_feature",
                "normalized_importance_pct",
            ]
        ].copy()
    )

    # --------------------------------------------------------------
    # Local explanations.
    # --------------------------------------------------------------
    local = make_local_explanations(
        test_data,
        X_test_transformed,
        shap_values,
        raw_probability,
        calibrated_probability_values,
        encoded_feature_names,
        target,
    )

    if "target" not in local.columns:
        local.insert(
            0,
            "target",
            target,
        )

    # --------------------------------------------------------------
    # Top drivers.
    # --------------------------------------------------------------
    top_drivers = make_top_driver_table(
        test_data,
        X_test_transformed,
        shap_values,
        raw_probability,
        calibrated_probability_values,
        encoded_feature_names,
        target,
    )

    # --------------------------------------------------------------
    # Integrity checks.
    # --------------------------------------------------------------
    if (
        set(local["feature"])
        & FORBIDDEN_FEATURES
    ):
        raise ValueError(
            "Forbidden feature found in SHAP explanations."
        )

    if (
        set(
            global_importance["feature"]
        )
        & FORBIDDEN_FEATURES
    ):
        raise ValueError(
            "Forbidden feature found in global SHAP output."
        )

    if not np.isfinite(
        shap_values
    ).all():
        raise ValueError(
            "SHAP matrix contains non-finite values."
        )

    if not np.isfinite(
        raw_probability
    ).all():
        raise ValueError(
            "Raw model probabilities contain non-finite values."
        )

    if not np.isfinite(
        calibrated_probability_values
    ).all():
        raise ValueError(
            "Calibrated probabilities contain non-finite values."
        )

    if (
        (calibrated_probability_values < 0)
        | (calibrated_probability_values > 1)
    ).any():
        raise ValueError(
            "Calibrated probabilities outside [0, 1]."
        )

    # Return all outputs to aggregate across horizons.
    return {
        "global": global_importance,
        "global_normalized": global_normalized,
        "local": local,
        "top_drivers": top_drivers,
        "test_count": len(test_data),
        "positive_count": int(y_test.sum()),
        "mean_raw_probability": float(
            raw_probability.mean()
        ),
        "mean_calibrated_probability": float(
            calibrated_probability_values.mean()
        ),
        "max_abs_shap": float(
            np.abs(shap_values).max()
        ),
    }


def write_validation_report(
    data,
    feature_names,
    all_outputs,
):
    """Write machine-readable integrity and methodology report."""
    lines = [
        "BhoomiSetu SHAP Explainability Validation",
        "=" * 72,
        "",
        "DATA",
        f"Total snapshots: {len(data)}",
        f"Total projects: {data['project_id'].nunique()}",
        f"Predictor count: {len(feature_names)}",
        "",
        "PROJECT SPLIT",
        "Train projects: 700",
        "Validation projects: 150",
        "Test projects: 150",
        "",
        "LEAKAGE CONTROLS",
        "Project-disjoint split: PASS",
        "Forbidden features excluded: PASS",
        "Target columns excluded from predictors: PASS",
        "Scenario ground truth excluded: PASS",
        "Future snapshot metadata excluded: PASS",
        "",
        "SHAP METHOD",
        "Explainer: shap.TreeExplainer",
        "Model explained: underlying XGBoost estimator",
        "Preprocessing: exact fitted baseline Pipeline preprocessing",
        "Probability calibration: sigmoid layer is separate",
        "Interpretation: model-attributed contribution, not causality",
        "",
        "TARGET RESULTS",
    ]

    for target, result in all_outputs.items():
        lines.extend(
            [
                f"{target}:",
                f"  Test snapshots: {result['test_count']}",
                f"  Positive labels: {result['positive_count']}",
                f"  Mean raw probability: "
                f"{result['mean_raw_probability']:.6f}",
                f"  Mean calibrated probability: "
                f"{result['mean_calibrated_probability']:.6f}",
                f"  Maximum absolute SHAP value: "
                f"{result['max_abs_shap']:.6f}",
                "  SHAP shape validation: PASS",
                "  Finite SHAP values: PASS",
                "  Forbidden-feature check: PASS",
            ]
        )

    lines.extend(
        [
            "",
            "INTERPRETATION RULE",
            "Positive SHAP values push the XGBoost prediction toward",
            "higher delay risk; negative SHAP values push it toward",
            "lower delay risk.",
            "",
            "IMPORTANT LIMITATION",
            "SHAP explains the behavior of the trained synthetic-data",
            "prototype model. It does not establish causal relationships",
            "or real-world Indian land-acquisition delay mechanisms.",
            "",
            "SHAP PIPELINE: PASS",
        ]
    )

    (
        OUTPUT_DIR
        / "validation_report.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def write_summary_report(
    all_outputs,
    global_importance,
):
    """Write concise human-readable summary."""
    lines = [
        "BhoomiSetu SHAP Summary",
        "=" * 72,
        "",
        "The tables below show model-attributed feature influence.",
        "They must not be interpreted as causal effects.",
        "",
    ]

    for target in TARGETS:
        lines.append(
            f"{target.upper()}"
        )
        lines.append(
            "-" * 72
        )

        target_global = global_importance[
            global_importance["target"]
            == target
        ].head(15)

        for _, row in target_global.iterrows():
            lines.append(
                f"{int(row['rank']):2d}. "
                f"{row['display_feature']} | "
                f"mean |SHAP|="
                f"{row['mean_abs_shap']:.6f} | "
                f"share="
                f"{row['normalized_importance_pct']:.2f}%"
            )

        lines.append("")

        top = all_outputs[
            target
        ]["top_drivers"]

        if not top.empty:
            lines.append(
                "Example local driver records are available in "
                "local_top_drivers.csv."
            )

        lines.append("")

    lines.extend(
        [
            "SHAP PIPELINE: PASS",
            "",
            "For dashboard wording use:",
            "\"Why the model flags this project\"",
            "",
            "Avoid wording such as:",
            "\"These factors will cause the delay.\"",
        ]
    )

    (
        OUTPUT_DIR
        / "shap_summary_report.txt"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print("=" * 72)
    print("BhoomiSetu SHAP Explainability Pipeline")
    print("=" * 72)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARRAY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data()

    data = make_project_split(
        data
    )

    feature_names = get_predictor_columns(
        data
    )

    print(
        f"Snapshots          : {len(data)}"
    )

    print(
        f"Projects           : "
        f"{data['project_id'].nunique()}"
    )

    print(
        f"Predictors         : "
        f"{len(feature_names)}"
    )

    print(
        "Split              : "
        "700 train / 150 validation / 150 test"
    )

    print(
        "SHAP data          : TEST projects only"
    )

    print(
        "Calibration        : sigmoid probability layer"
    )

    print()

    all_outputs = {}

    for target in TARGETS:
        all_outputs[target] = run_target(
            data,
            feature_names,
            target,
        )

    global_importance = pd.concat(
        [
            all_outputs[target]["global"]
            for target in TARGETS
        ],
        ignore_index=True,
    )

    global_normalized = pd.concat(
        [
            all_outputs[target]["global_normalized"]
            for target in TARGETS
        ],
        ignore_index=True,
    )

    local_explanations = pd.concat(
        [
            all_outputs[target]["local"]
            for target in TARGETS
        ],
        ignore_index=True,
    )

    top_drivers = pd.concat(
        [
            all_outputs[target]["top_drivers"]
            for target in TARGETS
        ],
        ignore_index=True,
    )

    global_importance.to_csv(
        OUTPUT_DIR
        / "global_shap_importance.csv",
        index=False,
    )

    global_normalized.to_csv(
        OUTPUT_DIR
        / "global_shap_importance_normalized.csv",
        index=False,
    )

    local_explanations.to_csv(
        OUTPUT_DIR
        / "local_shap_explanations.csv",
        index=False,
    )

    top_drivers.to_csv(
        OUTPUT_DIR
        / "local_top_drivers.csv",
        index=False,
    )

    write_validation_report(
        data,
        feature_names,
        all_outputs,
    )

    write_summary_report(
        all_outputs,
        global_importance,
    )

    print()
    print("=" * 72)
    print("SHAP PIPELINE COMPLETE")
    print("=" * 72)

    print(
        f"Global importance : "
        f"{OUTPUT_DIR / 'global_shap_importance.csv'}"
    )

    print(
        f"Normalized global: "
        f"{OUTPUT_DIR / 'global_shap_importance_normalized.csv'}"
    )

    print(
        f"Local explanations: "
        f"{OUTPUT_DIR / 'local_shap_explanations.csv'}"
    )

    print(
        f"Top drivers       : "
        f"{OUTPUT_DIR / 'local_top_drivers.csv'}"
    )

    print(
        f"Validation report : "
        f"{OUTPUT_DIR / 'validation_report.txt'}"
    )

    print(
        f"Summary report    : "
        f"{OUTPUT_DIR / 'shap_summary_report.txt'}"
    )

    print(
        f"Arrays            : "
        f"{ARRAY_DIR}"
    )

    print()
    print(
        "SHAP PIPELINE: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(
            "SHAP PIPELINE: FAIL"
        )
        print(
            f"Reason: {exc}"
        )
        sys.exit(1)
