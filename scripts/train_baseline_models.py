"""
BhoomiSetu Baseline ML Training Pipeline
========================================

Purpose
-------
Train and compare prototype delay classifiers for:
    delay_next_30d
    delay_next_60d
    delay_next_90d

Models
------
1. DummyClassifier (prior baseline)
2. LogisticRegression (interpretable baseline)
3. XGBoost (primary prototype model)

Split
-----
Project-disjoint chronological split using project_start_date:
    70% train / 15% validation / 15% test

IMPORTANT
---------
project_id and project_start_date are used ONLY for constructing the split.
They are never included in X.

The feature matrix in output/synthetic/ml/ml_features.csv intentionally
contains predictors + targets, but not project metadata. Therefore this
script reads project_id/project_start_date from synthetic_snapshots.csv and
aligns them to ml_features.csv by row order after validating exact row count
and target alignment.

Inputs
------
output/synthetic/ml/ml_features.csv
output/synthetic/synthetic_snapshots.csv

Outputs
-------
output/synthetic/ml/baseline/
    model_metrics.csv
    split_summary.csv
    predictions_test.csv
    feature_importance_xgboost_delay_next_30d.csv
    feature_importance_xgboost_delay_next_60d.csv
    feature_importance_xgboost_delay_next_90d.csv
    training_report.txt
    models/
        <target>_<model>.joblib

This is a prototype experiment on synthetic data. Metrics do not represent
real-world Indian land-acquisition performance.
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "output" / "synthetic" / "ml" / "ml_features.csv"
SNAPSHOT_FILE = ROOT / "output" / "synthetic" / "synthetic_snapshots.csv"

OUTPUT_DIR = ROOT / "output" / "synthetic" / "ml" / "baseline"
MODEL_DIR = OUTPUT_DIR / "models"

TARGETS = [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]

# Never use these as predictors.
FORBIDDEN_PREDICTORS = {
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

RANDOM_STATE = 20260914


def load_and_align_data():
    """Load feature matrix and attach project metadata safely."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature matrix not found:\n{FEATURE_FILE}\n"
            "Run scripts\\build_ml_features.py first."
        )

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(
            f"Snapshot dataset not found:\n{SNAPSHOT_FILE}"
        )

    features = pd.read_csv(FEATURE_FILE)
    snapshots = pd.read_csv(SNAPSHOT_FILE)

    if features.empty:
        raise ValueError("ML feature matrix is empty.")

    required_snapshot_cols = {
        "project_id",
        "project_start_date",
        *TARGETS,
    }
    missing_snapshot = sorted(
        required_snapshot_cols - set(snapshots.columns)
    )
    if missing_snapshot:
        raise ValueError(
            "synthetic_snapshots.csv is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing_snapshot)
        )

    # The feature builder preserves row order and writes one row per snapshot.
    # Validate this explicitly before carrying metadata across by row position.
    if len(features) != len(snapshots):
        raise ValueError(
            "Row-count mismatch between ml_features.csv and "
            "synthetic_snapshots.csv:\n"
            f"  features  = {len(features)}\n"
            f"  snapshots = {len(snapshots)}"
        )

    # Strong alignment check: target columns must match exactly.
    for target in TARGETS:
        a = pd.to_numeric(features[target], errors="coerce")
        b = pd.to_numeric(snapshots[target], errors="coerce")

        if not np.array_equal(
            a.fillna(-999999).to_numpy(),
            b.fillna(-999999).to_numpy(),
        ):
            raise ValueError(
                f"Target alignment check failed for {target}. "
                "Do not proceed until the feature and snapshot files "
                "are regenerated/aligned."
            )

    # Metadata comes from the original snapshot table, never from X.
    metadata = snapshots[
        ["project_id", "project_start_date"]
    ].copy()

    metadata["project_start_date"] = pd.to_datetime(
        metadata["project_start_date"],
        errors="coerce",
    )

    if metadata["project_start_date"].isna().any():
        raise ValueError(
            "Missing/invalid project_start_date found in snapshots."
        )

    data = features.copy()

    # Add metadata as non-predictor columns.
    data.insert(0, "project_id", metadata["project_id"].values)
    data.insert(
        1,
        "project_start_date",
        metadata["project_start_date"].values,
    )

    return data


def validate_project_structure(data):
    """Check project/snapshot structure."""
    project_counts = data.groupby("project_id").size()

    if project_counts.empty:
        raise ValueError("No projects found.")

    if data["project_id"].isna().any():
        raise ValueError("Missing project_id values found.")

    return {
        "projects": int(data["project_id"].nunique()),
        "snapshots": int(len(data)),
        "min_snapshots": int(project_counts.min()),
        "median_snapshots": float(project_counts.median()),
        "max_snapshots": int(project_counts.max()),
    }


def make_project_split(data):
    """
    Construct deterministic project-disjoint chronological split.

    Projects are sorted by project_start_date, then project_id.
    """
    project_table = (
        data[["project_id", "project_start_date"]]
        .drop_duplicates()
        .sort_values(
            ["project_start_date", "project_id"]
        )
        .reset_index(drop=True)
    )

    n = len(project_table)

    n_train = int(np.floor(n * 0.70))
    n_valid = int(np.floor(n * 0.15))
    n_test = n - n_train - n_valid

    if min(n_train, n_valid, n_test) <= 0:
        raise ValueError(
            "Not enough projects for a 70/15/15 split."
        )

    train_projects = set(
        project_table.iloc[:n_train]["project_id"]
    )

    valid_projects = set(
        project_table.iloc[
            n_train:n_train + n_valid
        ]["project_id"]
    )

    test_projects = set(
        project_table.iloc[
            n_train + n_valid:
        ]["project_id"]
    )

    if train_projects & valid_projects:
        raise ValueError("Train/validation project overlap detected.")

    if train_projects & test_projects:
        raise ValueError("Train/test project overlap detected.")

    if valid_projects & test_projects:
        raise ValueError("Validation/test project overlap detected.")

    assigned_projects = (
        train_projects | valid_projects | test_projects
    )

    if len(assigned_projects) != n:
        raise ValueError(
            "Some projects were not assigned to a split."
        )

    result = data.copy()

    result["split"] = "unassigned"
    result.loc[
        result["project_id"].isin(train_projects),
        "split",
    ] = "train"

    result.loc[
        result["project_id"].isin(valid_projects),
        "split",
    ] = "validation"

    result.loc[
        result["project_id"].isin(test_projects),
        "split",
    ] = "test"

    if (result["split"] == "unassigned").any():
        raise ValueError(
            "Some snapshots were not assigned to a split."
        )

    # Explicit row-level verification.
    split_counts = (
        result.groupby("project_id")["split"]
        .nunique()
    )

    leaked_projects = split_counts[
        split_counts > 1
    ]

    if not leaked_projects.empty:
        raise ValueError(
            "Project leakage detected: some projects occur in "
            "multiple splits."
        )

    return (
        result,
        project_table,
        train_projects,
        valid_projects,
        test_projects,
    )


def choose_feature_columns(data):
    """Identify predictor columns and ensure no forbidden fields enter X."""
    feature_columns = [
        c for c in data.columns
        if c not in FORBIDDEN_PREDICTORS
        and c not in {
            "split",
            "project_start_date",
            "project_id",
        }
    ]

    if not feature_columns:
        raise ValueError("No predictor features remain.")

    forbidden_found = sorted(
        set(feature_columns) & FORBIDDEN_PREDICTORS
    )

    if forbidden_found:
        raise ValueError(
            "Forbidden columns found in predictor matrix:\n"
            + "\n".join(f"  - {c}" for c in forbidden_found)
        )

    numeric = [
        c for c in feature_columns
        if pd.api.types.is_numeric_dtype(data[c])
    ]

    categorical = [
        c for c in feature_columns
        if c not in numeric
    ]

    return feature_columns, numeric, categorical


def make_preprocessor(numeric_features, categorical_features):
    """Create consistent preprocessing for train/validation/test."""
    numeric_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipe,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipe,
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def safe_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(
        average_precision_score(y_true, y_prob)
    )


def precision_at_k(y_true, y_prob, fraction=0.10):
    """Precision in the highest-risk fraction."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return np.nan

    k = max(
        1,
        int(np.ceil(len(y_true) * fraction)),
    )

    order = np.argsort(-y_prob)
    top = y_true[order[:k]]

    return float(np.mean(top))


def recall_at_k(y_true, y_prob, fraction=0.10):
    """Fraction of all positives captured in the highest-risk fraction."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    positives = int(y_true.sum())

    if positives == 0:
        return np.nan

    k = max(
        1,
        int(np.ceil(len(y_true) * fraction)),
    )

    order = np.argsort(-y_prob)
    captured = int(
        y_true[order[:k]].sum()
    )

    return float(captured / positives)


def evaluate_predictions(
    y_true,
    y_prob,
    threshold=0.50,
):
    """Compute classification and ranking metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    return {
        "roc_auc": safe_auc(
            y_true,
            y_prob,
        ),
        "pr_auc": safe_pr_auc(
            y_true,
            y_prob,
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                y_prob,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "precision_at_10pct": precision_at_k(
            y_true,
            y_prob,
            0.10,
        ),
        "recall_at_10pct": recall_at_k(
            y_true,
            y_prob,
            0.10,
        ),
        "positive_rate": float(
            np.mean(y_true)
        ),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def train_dummy(X_train, y_train):
    """Prior-probability baseline."""
    model = DummyClassifier(
        strategy="prior"
    )

    # DummyClassifier only needs a placeholder matrix.
    placeholder = np.zeros(
        (len(X_train), 1),
        dtype=float,
    )

    model.fit(
        placeholder,
        y_train,
    )

    return model


def train_logistic(
    preprocessor,
    X_train,
    y_train,
):
    """Train interpretable logistic baseline."""
    model = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


def train_xgboost(
    preprocessor,
    X_train,
    y_train,
):
    """Train primary XGBoost prototype."""
    positives = max(
        int(y_train.sum()),
        1,
    )

    negatives = max(
        int(len(y_train) - y_train.sum()),
        1,
    )

    scale_pos_weight = (
        negatives / positives
    )

    model = Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor,
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=3,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )

    model.fit(
        X_train,
        y_train,
    )

    return model, scale_pos_weight


def save_xgb_feature_importance(
    model,
    output_file,
    target,
):
    """Save XGBoost gain-style feature importance after preprocessing."""
    try:
        preprocessor = model.named_steps[
            "preprocess"
        ]

        xgb = model.named_steps["model"]

        names = list(
            preprocessor.get_feature_names_out()
        )

        importance = xgb.feature_importances_

        if len(names) != len(importance):
            print(
                "WARNING: Feature-name/importance length mismatch "
                f"for {target}; importance file skipped."
            )
            return

        result = pd.DataFrame(
            {
                "target": target,
                "feature": names,
                "importance": importance,
            }
        ).sort_values(
            "importance",
            ascending=False,
        )

        result.to_csv(
            output_file,
            index=False,
        )

    except Exception as exc:
        print(
            f"WARNING: Could not save feature importance for "
            f"{target}: {exc}"
        )


def main():
    print("=" * 72)
    print("BhoomiSetu Baseline ML Training")
    print("=" * 72)
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Load + align
    # --------------------------------------------------------------
    data = load_and_align_data()

    structure = validate_project_structure(
        data
    )

    (
        data,
        project_table,
        train_projects,
        valid_projects,
        test_projects,
    ) = make_project_split(data)

    (
        feature_columns,
        numeric_features,
        categorical_features,
    ) = choose_feature_columns(data)

    print(
        f"Snapshots          : {structure['snapshots']}"
    )
    print(
        f"Projects           : {structure['projects']}"
    )
    print(
        f"Snapshots/project  : "
        f"{structure['min_snapshots']} min / "
        f"{structure['median_snapshots']:.1f} median / "
        f"{structure['max_snapshots']} max"
    )
    print(
        f"Predictor features : {len(feature_columns)}"
    )
    print(
        f"  Numeric          : {len(numeric_features)}"
    )
    print(
        f"  Categorical      : {len(categorical_features)}"
    )
    print()

    print(
        "Project split      : "
        f"{len(train_projects)} train / "
        f"{len(valid_projects)} validation / "
        f"{len(test_projects)} test"
    )

    print(
        "Project leakage    : NONE"
    )

    print()

    # --------------------------------------------------------------
    # Split summary
    # --------------------------------------------------------------
    split_summary = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        part = data[
            data["split"] == split_name
        ]

        split_summary.append(
            {
                "split": split_name,
                "projects": int(
                    part["project_id"].nunique()
                ),
                "snapshots": int(len(part)),
                "earliest_project_start": (
                    part["project_start_date"].min()
                ),
                "latest_project_start": (
                    part["project_start_date"].max()
                ),
            }
        )

    split_summary_df = pd.DataFrame(
        split_summary
    )

    split_summary_df.to_csv(
        OUTPUT_DIR / "split_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Train models for each horizon
    # --------------------------------------------------------------
    results = []
    prediction_frames = []

    for target in TARGETS:
        print()
        print("-" * 72)
        print(f"TARGET: {target}")
        print("-" * 72)

        train = data[
            data["split"] == "train"
        ].copy()

        valid = data[
            data["split"] == "validation"
        ].copy()

        test = data[
            data["split"] == "test"
        ].copy()

        X_train = train[
            feature_columns
        ]
        y_train = train[
            target
        ].astype(int)

        X_valid = valid[
            feature_columns
        ]
        y_valid = valid[
            target
        ].astype(int)

        X_test = test[
            feature_columns
        ]
        y_test = test[
            target
        ].astype(int)

        print(
            f"Train : {int(y_train.sum())}/{len(y_train)} "
            f"positive ({y_train.mean()*100:.2f}%)"
        )

        print(
            f"Valid : {int(y_valid.sum())}/{len(y_valid)} "
            f"positive ({y_valid.mean()*100:.2f}%)"
        )

        print(
            f"Test  : {int(y_test.sum())}/{len(y_test)} "
            f"positive ({y_test.mean()*100:.2f}%)"
        )

        # Validation is evaluated here for diagnostics only.
        # We do NOT use validation results to modify hyperparameters
        # automatically in this script.
        models = {}

        dummy = train_dummy(
            X_train,
            y_train,
        )

        models["dummy"] = dummy

        logistic = train_logistic(
            make_preprocessor(
                numeric_features,
                categorical_features,
            ),
            X_train,
            y_train,
        )

        models["logistic_regression"] = logistic

        xgb, scale_pos_weight = train_xgboost(
            make_preprocessor(
                numeric_features,
                categorical_features,
            ),
            X_train,
            y_train,
        )

        models["xgboost"] = xgb

        # ----------------------------------------------------------
        # Validation + test evaluation
        # ----------------------------------------------------------
        for model_name, model in models.items():

            if model_name == "dummy":
                valid_prob = model.predict_proba(
                    np.zeros(
                        (len(X_valid), 1)
                    )
                )[:, 1]

                test_prob = model.predict_proba(
                    np.zeros(
                        (len(X_test), 1)
                    )
                )[:, 1]

            else:
                valid_prob = model.predict_proba(
                    X_valid
                )[:, 1]

                test_prob = model.predict_proba(
                    X_test
                )[:, 1]

            valid_metrics = evaluate_predictions(
                y_valid,
                valid_prob,
            )

            test_metrics = evaluate_predictions(
                y_test,
                test_prob,
            )

            results.append(
                {
                    "target": target,
                    "model": model_name,
                    "split": "validation",
                    **valid_metrics,
                    "scale_pos_weight": (
                        float(scale_pos_weight)
                        if model_name == "xgboost"
                        else np.nan
                    ),
                }
            )

            results.append(
                {
                    "target": target,
                    "model": model_name,
                    "split": "test",
                    **test_metrics,
                    "scale_pos_weight": (
                        float(scale_pos_weight)
                        if model_name == "xgboost"
                        else np.nan
                    ),
                }
            )

            prediction_frames.append(
                pd.DataFrame(
                    {
                        "project_id": test[
                            "project_id"
                        ].values,
                        "target": target,
                        "model": model_name,
                        "y_true": y_test.values,
                        "predicted_probability": test_prob,
                    }
                )
            )

            print()
            print(
                f"{model_name:20s}"
            )
            print(
                f"  Validation "
                f"ROC-AUC={valid_metrics['roc_auc']:.4f}  "
                f"PR-AUC={valid_metrics['pr_auc']:.4f}  "
                f"Brier={valid_metrics['brier_score']:.4f}"
            )
            print(
                f"  Test       "
                f"ROC-AUC={test_metrics['roc_auc']:.4f}  "
                f"PR-AUC={test_metrics['pr_auc']:.4f}  "
                f"Brier={test_metrics['brier_score']:.4f}  "
                f"Recall={test_metrics['recall']:.4f}  "
                f"P@10={test_metrics['precision_at_10pct']:.4f}"
            )

            model_file = (
                MODEL_DIR
                / f"{target}_{model_name}.joblib"
            )

            joblib.dump(
                model,
                model_file,
            )

        importance_file = (
            OUTPUT_DIR
            / f"feature_importance_xgboost_{target}.csv"
        )

        save_xgb_feature_importance(
            xgb,
            importance_file,
            target,
        )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------
    results_df = pd.DataFrame(results)

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    results_df.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "predictions_test.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------
    report = [
        "BhoomiSetu Baseline ML Training Report",
        "=" * 72,
        "",
        "DATA",
        f"Snapshots: {len(data)}",
        f"Projects: {data['project_id'].nunique()}",
        f"Predictors: {len(feature_columns)}",
        f"Numeric predictors: {len(numeric_features)}",
        f"Categorical predictors: {len(categorical_features)}",
        "",
        "PROJECT-DISJOINT CHRONOLOGICAL SPLIT",
        f"Train projects: {len(train_projects)}",
        f"Validation projects: {len(valid_projects)}",
        f"Test projects: {len(test_projects)}",
        "",
        "LEAKAGE CHECKS",
        "Project overlap train/validation: 0",
        "Project overlap train/test: 0",
        "Project overlap validation/test: 0",
        "Project metadata used only for split: YES",
        "Future target columns used as predictors: NO",
        "Scenario ground truth used as predictor: NO",
        "Snapshot identifiers used as predictors: NO",
        "",
        "MODELS",
        "DummyClassifier(strategy='prior')",
        "LogisticRegression(class_weight='balanced')",
        "XGBClassifier",
        "",
        "INTERPRETATION",
        "Validation results are diagnostic only.",
        "The test set is not used to tune the models.",
        "This is a prototype experiment using synthetic data.",
        "Metrics do not represent real-world Indian land-acquisition accuracy.",
        "Synthetic scenario labels are not model features.",
        "",
        "RESULTS",
    ]

    for _, row in results_df.iterrows():
        report.append(
            f"{row['target']} | "
            f"{row['model']} | "
            f"{row['split']} | "
            f"ROC-AUC={row['roc_auc']:.6f} | "
            f"PR-AUC={row['pr_auc']:.6f} | "
            f"Brier={row['brier_score']:.6f} | "
            f"Recall={row['recall']:.6f} | "
            f"P@10={row['precision_at_10pct']:.6f}"
        )

    report.extend(
        [
            "",
            "BASELINE TRAINING: PASS",
        ]
    )

    (
        OUTPUT_DIR / "training_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 72)
    print(
        "Metrics            : "
        f"{OUTPUT_DIR / 'model_metrics.csv'}"
    )
    print(
        "Test predictions   : "
        f"{OUTPUT_DIR / 'predictions_test.csv'}"
    )
    print(
        "Split summary      : "
        f"{OUTPUT_DIR / 'split_summary.csv'}"
    )
    print(
        "Report             : "
        f"{OUTPUT_DIR / 'training_report.txt'}"
    )
    print(
        "Models             : "
        f"{MODEL_DIR}"
    )
    print()
    print("BASELINE TRAINING: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("BASELINE TRAINING: FAIL")
        print(f"Reason: {exc}")
        sys.exit(1)
