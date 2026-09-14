"""
BhoomiSetu Probability Calibration Pipeline
============================================

Calibrates already-trained XGBoost delay models using validation-project
predictions, then evaluates the calibration on untouched test projects.

Targets:
    delay_next_30d
    delay_next_60d
    delay_next_90d

Methods:
    1. Uncalibrated XGBoost
    2. Sigmoid / Platt-style calibration
    3. Isotonic calibration

Protocol:
    TRAIN projects:
        already used to fit the XGBoost models.

    VALIDATION projects:
        used ONLY to learn calibration mappings.

    TEST projects:
        used ONLY for final evaluation.

No test labels are used to fit calibration.

This implementation deliberately does not use cv="prefit", because recent
scikit-learn versions removed that API. Calibration mappings are fitted
explicitly from validation probabilities.

The sigmoid artifact is saved as a plain dictionary containing the fitted
LogisticRegression coefficient, intercept, epsilon, method, and version.
This makes it safe to load from other scripts without depending on the
custom SigmoidCalibrator class being importable from __main__.

Inputs:
    output/synthetic/ml/ml_features.csv
    output/synthetic/synthetic_snapshots.csv
    output/synthetic/ml/baseline/models/
        delay_next_XXd_xgboost.joblib

Outputs:
    output/synthetic/ml/calibration/
        calibration_metrics.csv
        calibration_bins.csv
        calibrated_test_predictions.csv
        calibration_report.txt
        models/
            <target>_sigmoid.joblib   # portable dict: coefficient/intercept
            <target>_isotonic.joblib
"""

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

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

OUTPUT_DIR = (
    ROOT
    / "output"
    / "synthetic"
    / "ml"
    / "calibration"
)

MODEL_DIR = OUTPUT_DIR / "models"

TARGETS = [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]

RANDOM_STATE = 20260914


def load_and_align_data():
    """Load feature data and safely attach project metadata."""
    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature matrix not found:\n{FEATURE_FILE}"
        )

    if not SNAPSHOT_FILE.exists():
        raise FileNotFoundError(
            f"Snapshot dataset not found:\n{SNAPSHOT_FILE}"
        )

    features = pd.read_csv(FEATURE_FILE)
    snapshots = pd.read_csv(SNAPSHOT_FILE)

    if len(features) != len(snapshots):
        raise ValueError(
            "Feature/snapshot row-count mismatch: "
            f"{len(features)} vs {len(snapshots)}"
        )

    required = {
        "project_id",
        "project_start_date",
        *TARGETS,
    }

    missing = sorted(
        required - set(snapshots.columns)
    )

    if missing:
        raise ValueError(
            "Snapshot dataset is missing:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    # Validate that feature targets still align with the original snapshots.
    for target in TARGETS:
        feature_target = pd.to_numeric(
            features[target],
            errors="coerce",
        )

        snapshot_target = pd.to_numeric(
            snapshots[target],
            errors="coerce",
        )

        if not np.array_equal(
            feature_target.fillna(-999999).to_numpy(),
            snapshot_target.fillna(-999999).to_numpy(),
        ):
            raise ValueError(
                f"Target alignment failed for {target}."
            )

    metadata = snapshots[
        [
            "project_id",
            "project_start_date",
        ]
    ].copy()

    metadata["project_start_date"] = pd.to_datetime(
        metadata["project_start_date"],
        errors="coerce",
    )

    if metadata["project_start_date"].isna().any():
        raise ValueError(
            "Invalid or missing project_start_date."
        )

    data = features.copy()

    # Metadata is used for splitting only.
    data.insert(
        0,
        "project_id",
        metadata["project_id"].values,
    )

    data.insert(
        1,
        "project_start_date",
        metadata["project_start_date"].values,
    )

    return data


def make_project_split(data):
    """Reproduce the exact 70/15/15 chronological project split."""
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

    n = len(project_table)

    n_train = int(
        np.floor(n * 0.70)
    )

    n_valid = int(
        np.floor(n * 0.15)
    )

    train_projects = set(
        project_table.iloc[
            :n_train
        ]["project_id"]
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
        raise ValueError(
            "Train/validation project overlap."
        )

    if train_projects & test_projects:
        raise ValueError(
            "Train/test project overlap."
        )

    if valid_projects & test_projects:
        raise ValueError(
            "Validation/test project overlap."
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

    if (
        result["split"] == "unassigned"
    ).any():
        raise ValueError(
            "Some snapshots were not assigned."
        )

    # Final row-level leakage verification.
    split_counts = (
        result.groupby("project_id")["split"]
        .nunique()
    )

    if (split_counts > 1).any():
        raise ValueError(
            "Project leakage detected."
        )

    return result


def choose_features(data):
    """Return predictor columns exactly as used by baseline training."""
    forbidden = {
        "snapshot_id",
        "project_id",
        "snapshot_date",
        "project_start_date",
        "stage_entry_date",
        "is_censored",
        "scenario",
        "scenario_ground_truth",
        "event_count_so_far",
        "split",
        *TARGETS,
    }

    features = [
        c for c in data.columns
        if c not in forbidden
    ]

    if not features:
        raise ValueError(
            "No predictor features found."
        )

    if set(features) & forbidden:
        raise ValueError(
            "Forbidden predictor detected."
        )

    return features


def expected_calibration_error(
    y_true,
    probabilities,
    n_bins=10,
):
    """Compute equal-width 10-bin Expected Calibration Error."""
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(y_true) == 0:
        return np.nan

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    ece = 0.0

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            continue

        predicted = float(
            probabilities[mask].mean()
        )

        observed = float(
            y_true[mask].mean()
        )

        ece += (
            count / len(y_true)
        ) * abs(
            predicted - observed
        )

    return float(ece)


def calibration_bin_table(
    y_true,
    probabilities,
    n_bins=10,
):
    """Return reliability-bin statistics."""
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    rows = []

    for i in range(n_bins):
        lower = edges[i]
        upper = edges[i + 1]

        if i == n_bins - 1:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        count = int(mask.sum())

        if count == 0:
            rows.append(
                {
                    "bin": i + 1,
                    "lower_probability": lower,
                    "upper_probability": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_positive_rate": np.nan,
                    "absolute_calibration_gap": np.nan,
                }
            )
            continue

        predicted = float(
            probabilities[mask].mean()
        )

        observed = float(
            y_true[mask].mean()
        )

        rows.append(
            {
                "bin": i + 1,
                "lower_probability": lower,
                "upper_probability": upper,
                "count": count,
                "mean_predicted_probability": predicted,
                "observed_positive_rate": observed,
                "absolute_calibration_gap": abs(
                    predicted - observed
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate(y_true, probabilities):
    """Calculate discrimination and probability-quality metrics."""
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    return {
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        "ece_10bin": float(
            expected_calibration_error(
                y_true,
                probabilities,
                10,
            )
        ),
    }


class SigmoidCalibrator:
    """
    Platt-style probability calibration.

    Fits logistic regression on the logit of the base-model probability.
    A small epsilon prevents log(0) and log(1).

    The learned LogisticRegression is the complete calibration model.
    """

    def __init__(self):
        self.epsilon = 1e-6
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    def _logit(self, probabilities):
        probabilities = np.clip(
            np.asarray(
                probabilities,
                dtype=float,
            ),
            self.epsilon,
            1.0 - self.epsilon,
        )

        return np.log(
            probabilities
            / (1.0 - probabilities)
        ).reshape(-1, 1)

    def fit(self, probabilities, y_true):
        x = self._logit(probabilities)

        self.model.fit(
            x,
            np.asarray(y_true).astype(int),
        )

        return self

    def predict_proba(self, probabilities):
        x = self._logit(probabilities)

        return self.model.predict_proba(x)[:, 1]


def fit_isotonic(
    validation_probabilities,
    y_valid,
):
    """Fit monotonic isotonic mapping on validation predictions."""
    model = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    model.fit(
        np.asarray(
            validation_probabilities,
            dtype=float,
        ),
        np.asarray(
            y_valid,
            dtype=int,
        ),
    )

    return model


def fit_sigmoid(
    validation_probabilities,
    y_valid,
):
    """Fit Platt-style sigmoid calibration."""
    calibrator = SigmoidCalibrator()

    calibrator.fit(
        validation_probabilities,
        y_valid,
    )

    return calibrator


def main():
    print("=" * 72)
    print("BhoomiSetu Probability Calibration")
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

    data = load_and_align_data()

    data = make_project_split(
        data
    )

    feature_columns = choose_features(
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
        f"{len(feature_columns)}"
    )

    print(
        "Split              : "
        "700 train / 150 validation / 150 test"
    )

    print(
        "Calibration source : validation projects ONLY"
    )

    print(
        "Final evaluation   : test projects ONLY"
    )

    print(
        "Calibration API    : explicit mapping "
        "(no cv='prefit')"
    )

    print()

    results = []
    prediction_frames = []
    calibration_bins = []

    for target in TARGETS:
        print("-" * 72)
        print(
            f"TARGET: {target}"
        )
        print("-" * 72)

        baseline_model_file = (
            BASELINE_MODEL_DIR
            / f"{target}_xgboost.joblib"
        )

        if not baseline_model_file.exists():
            raise FileNotFoundError(
                "Baseline XGBoost model not found:\n"
                f"{baseline_model_file}\n"
                "Run train_baseline_models.py first."
            )

        base_model = joblib.load(
            baseline_model_file
        )

        validation = data[
            data["split"] == "validation"
        ]

        test = data[
            data["split"] == "test"
        ]

        X_valid = validation[
            feature_columns
        ]

        y_valid = validation[
            target
        ].astype(int)

        X_test = test[
            feature_columns
        ]

        y_test = test[
            target
        ].astype(int)

        # ----------------------------------------------------------
        # Base XGBoost probabilities.
        # ----------------------------------------------------------
        valid_prob = (
            base_model.predict_proba(
                X_valid
            )[:, 1]
        )

        test_prob = (
            base_model.predict_proba(
                X_test
            )[:, 1]
        )

        # ----------------------------------------------------------
        # Uncalibrated baseline.
        # ----------------------------------------------------------
        raw_metrics = evaluate(
            y_test,
            test_prob,
        )

        results.append(
            {
                "target": target,
                "method": "uncalibrated_xgboost",
                "split": "test",
                **raw_metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "project_id": test[
                        "project_id"
                    ].values,
                    "target": target,
                    "method": "uncalibrated_xgboost",
                    "y_true": y_test.values,
                    "predicted_probability": test_prob,
                }
            )
        )

        raw_bins = calibration_bin_table(
            y_test,
            test_prob,
        )

        raw_bins.insert(
            0,
            "method",
            "uncalibrated_xgboost",
        )

        raw_bins.insert(
            0,
            "target",
            target,
        )

        calibration_bins.append(
            raw_bins
        )

        print(
            f"uncalibrated_xgboost "
            f"ROC-AUC={raw_metrics['roc_auc']:.4f}  "
            f"PR-AUC={raw_metrics['pr_auc']:.4f}  "
            f"Brier={raw_metrics['brier_score']:.4f}  "
            f"ECE={raw_metrics['ece_10bin']:.4f}"
        )

        # ----------------------------------------------------------
        # Sigmoid calibration.
        #
        # IMPORTANT:
        # The calibration model sees ONLY validation predictions
        # and validation labels.
        # ----------------------------------------------------------
        sigmoid = fit_sigmoid(
            valid_prob,
            y_valid,
        )

        sigmoid_test_prob = (
            sigmoid.predict_proba(
                test_prob
            )
        )

        sigmoid_metrics = evaluate(
            y_test,
            sigmoid_test_prob,
        )

        results.append(
            {
                "target": target,
                "method": "sigmoid",
                "split": "test",
                **sigmoid_metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "project_id": test[
                        "project_id"
                    ].values,
                    "target": target,
                    "method": "sigmoid",
                    "y_true": y_test.values,
                    "predicted_probability": sigmoid_test_prob,
                }
            )
        )

        sigmoid_bins = calibration_bin_table(
            y_test,
            sigmoid_test_prob,
        )

        sigmoid_bins.insert(
            0,
            "method",
            "sigmoid",
        )

        sigmoid_bins.insert(
            0,
            "target",
            target,
        )

        calibration_bins.append(
            sigmoid_bins
        )

        # Save a portable calibration artifact instead of the custom
        # SigmoidCalibrator object. This avoids joblib depending on
        # __main__.SigmoidCalibrator when another script loads it.
        sigmoid_artifact = {
            "method": "sigmoid",
            "version": 1,
            "epsilon": float(sigmoid.epsilon),
            "coefficient": float(
                np.asarray(sigmoid.model.coef_).reshape(-1)[0]
            ),
            "intercept": float(
                np.asarray(sigmoid.model.intercept_).reshape(-1)[0]
            ),
        }

        joblib.dump(
            sigmoid_artifact,
            MODEL_DIR
            / f"{target}_sigmoid.joblib",
        )

        print(
            f"{'sigmoid':20s} "
            f"ROC-AUC={sigmoid_metrics['roc_auc']:.4f}  "
            f"PR-AUC={sigmoid_metrics['pr_auc']:.4f}  "
            f"Brier={sigmoid_metrics['brier_score']:.4f}  "
            f"ECE={sigmoid_metrics['ece_10bin']:.4f}"
        )

        # ----------------------------------------------------------
        # Isotonic calibration.
        # ----------------------------------------------------------
        isotonic = fit_isotonic(
            valid_prob,
            y_valid,
        )

        isotonic_test_prob = np.asarray(
            isotonic.predict(
                test_prob
            ),
            dtype=float,
        )

        isotonic_metrics = evaluate(
            y_test,
            isotonic_test_prob,
        )

        results.append(
            {
                "target": target,
                "method": "isotonic",
                "split": "test",
                **isotonic_metrics,
            }
        )

        prediction_frames.append(
            pd.DataFrame(
                {
                    "project_id": test[
                        "project_id"
                    ].values,
                    "target": target,
                    "method": "isotonic",
                    "y_true": y_test.values,
                    "predicted_probability": isotonic_test_prob,
                }
            )
        )

        isotonic_bins = calibration_bin_table(
            y_test,
            isotonic_test_prob,
        )

        isotonic_bins.insert(
            0,
            "method",
            "isotonic",
        )

        isotonic_bins.insert(
            0,
            "target",
            target,
        )

        calibration_bins.append(
            isotonic_bins
        )

        joblib.dump(
            isotonic,
            MODEL_DIR
            / f"{target}_isotonic.joblib",
        )

        print(
            f"{'isotonic':20s} "
            f"ROC-AUC={isotonic_metrics['roc_auc']:.4f}  "
            f"PR-AUC={isotonic_metrics['pr_auc']:.4f}  "
            f"Brier={isotonic_metrics['brier_score']:.4f}  "
            f"ECE={isotonic_metrics['ece_10bin']:.4f}"
        )

    results_df = pd.DataFrame(
        results
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    bins_df = pd.concat(
        calibration_bins,
        ignore_index=True,
    )

    results_df.to_csv(
        OUTPUT_DIR
        / "calibration_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR
        / "calibrated_test_predictions.csv",
        index=False,
    )

    bins_df.to_csv(
        OUTPUT_DIR
        / "calibration_bins.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Human-readable report.
    # --------------------------------------------------------------
    report = [
        "BhoomiSetu Probability Calibration Report",
        "=" * 72,
        "",
        "DATA",
        f"Snapshots: {len(data)}",
        f"Projects: {data['project_id'].nunique()}",
        f"Predictors: {len(feature_columns)}",
        "",
        "SPLIT",
        "Train projects: 700",
        "Validation/calibration projects: 150",
        "Test projects: 150",
        "",
        "CALIBRATION PROTOCOL",
        "1. XGBoost was previously trained on TRAIN projects.",
        "2. XGBoost validation probabilities were generated.",
        "3. Sigmoid and isotonic mappings were fitted using validation",
        "   probabilities and validation labels only.",
        "4. Calibrated probabilities were generated for TEST projects.",
        "5. TEST labels were used only for final evaluation.",
        "",
        "LEAKAGE CHECKS",
        "Project-disjoint split: PASS",
        "Calibration fitted on test labels: NO",
        "Future target used as predictor: NO",
        "Scenario ground truth used: NO",
        "",
        "RESULTS",
    ]

    for _, row in results_df.iterrows():
        report.append(
            f"{row['target']} | "
            f"{row['method']} | "
            f"ROC-AUC={row['roc_auc']:.6f} | "
            f"PR-AUC={row['pr_auc']:.6f} | "
            f"Brier={row['brier_score']:.6f} | "
            f"ECE={row['ece_10bin']:.6f}"
        )

    report.extend(
        [
            "",
            "INTERPRETATION",
            "Lower Brier score is better.",
            "Lower ECE is better calibrated.",
            "ROC-AUC and PR-AUC measure ranking/discrimination.",
            "Calibration can change probability quality without necessarily",
            "improving ranking metrics.",
            "",
            "IMPORTANT LIMITATION",
            "All results are based on synthetic prototype data.",
            "They do not establish real-world Indian land-acquisition",
            "delay probabilities or production accuracy.",
            "",
            "CALIBRATION PIPELINE: PASS",
        ]
    )

    (
        OUTPUT_DIR
        / "calibration_report.txt"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("CALIBRATION COMPLETE")
    print("=" * 72)

    print(
        "Metrics      : "
        f"{OUTPUT_DIR / 'calibration_metrics.csv'}"
    )

    print(
        "Bins         : "
        f"{OUTPUT_DIR / 'calibration_bins.csv'}"
    )

    print(
        "Predictions  : "
        f"{OUTPUT_DIR / 'calibrated_test_predictions.csv'}"
    )

    print(
        "Report       : "
        f"{OUTPUT_DIR / 'calibration_report.txt'}"
    )

    print(
        "Models       : "
        f"{MODEL_DIR}"
    )

    print()
    print(
        "CALIBRATION PIPELINE: PASS"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print(
            "CALIBRATION PIPELINE: FAIL"
        )
        print(
            f"Reason: {exc}"
        )
        sys.exit(1)
