"""
BhoomiSetu ML Feature Builder
==============================

Builds the reproducible feature matrix used by BhoomiSetu prototype ML models.

Input:
    output/synthetic/synthetic_snapshots.csv

Outputs:
    output/synthetic/ml/
        ml_features.csv
        feature_manifest.csv
        feature_build_report.txt

Design principles:
- Never use future outcome labels as model inputs.
- Never use project/snapshot identifiers as predictors.
- Never use scenario ground truth as a predictor.
- Preserve the three forward-looking targets separately.
- Create only features that are available at the snapshot time.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "output" / "synthetic" / "synthetic_snapshots.csv"
OUTPUT_DIR = ROOT / "output" / "synthetic" / "ml"

TARGETS = [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]

# Columns that must never be model predictors.
FORBIDDEN = {
    "snapshot_id",
    "project_id",
    "snapshot_date",
    "project_start_date",
    "stage_entry_date",
    "is_censored",
    "scenario",
    "scenario_ground_truth",
    *TARGETS,
}

# Current candidate feature set from the ML-readiness audit.
BASE_NUMERIC = [
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
]

BASE_CATEGORICAL = [
    "current_stage",
    "approval_dept",
    "terrain_type",
    "urban_rural",
]


def safe_divide(numerator, denominator):
    """Element-wise division; returns NaN where denominator is <= 0."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def validate_input(df):
    required = set(BASE_NUMERIC + BASE_CATEGORICAL + TARGETS)
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Input dataset is missing required columns:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )

    forbidden_present = sorted(FORBIDDEN & set(df.columns))
    if not forbidden_present:
        raise ValueError("Expected leakage/target columns were not found in input.")

    if len(df) == 0:
        raise ValueError("Input dataset is empty.")


def build_features(df):
    out = df.copy()

    # ------------------------------------------------------------------
    # Remove explicitly forbidden columns from the predictor matrix.
    # Targets are restored at the end so the output remains one
    # reproducible modeling table.
    # ------------------------------------------------------------------
    target_values = out[TARGETS].copy()

    # Snapshot timing: event_count_so_far is intentionally excluded
    # because it is perfectly redundant with days_since_project_start
    # under the 30-day snapshot cadence.
    if "event_count_so_far" in out.columns:
        out = out.drop(columns=["event_count_so_far"])

    # ------------------------------------------------------------------
    # Engineered features
    # ------------------------------------------------------------------

    # Fraction of required land that remains to be acquired.
    out["land_acquisition_ratio"] = safe_divide(
        out["land_to_acquire_ha"], out["land_required_ha"]
    )

    # Family burden relative to parcel count.
    out["affected_family_per_parcel"] = safe_divide(
        out["affected_families"], out["total_parcels"]
    )

    # Vulnerable-family share among affected families.
    out["vulnerable_family_share"] = safe_divide(
        out["vulnerable_families"], out["affected_families"]
    )

    # Absolute compensation still not disbursed.
    out["compensation_pending_cr"] = (
        pd.to_numeric(out["compensation_awarded_cr"], errors="coerce")
        - pd.to_numeric(out["compensation_disbursed_cr"], errors="coerce")
    ).clip(lower=0)

    # Remaining R&R progress.
    out["rnr_remaining_pct"] = (
        100 - pd.to_numeric(out["rnr_progress_pct"], errors="coerce")
    ).clip(lower=0, upper=100)

    # Remaining documentation completion.
    out["docs_remaining_pct"] = (
        100 - pd.to_numeric(out["docs_complete_pct"], errors="coerce")
    ).clip(lower=0, upper=100)

    # ------------------------------------------------------------------
    # Predictor list
    # ------------------------------------------------------------------
    engineered_numeric = [
        "land_acquisition_ratio",
        "affected_family_per_parcel",
        "vulnerable_family_share",
        "compensation_pending_cr",
        "rnr_remaining_pct",
        "docs_remaining_pct",
    ]

    numeric_features = [
        c for c in BASE_NUMERIC if c != "event_count_so_far"
    ] + engineered_numeric

    categorical_features = BASE_CATEGORICAL.copy()

    feature_columns = numeric_features + categorical_features

    # Ensure no forbidden feature slipped into the matrix.
    leakage_features = sorted(set(feature_columns) & FORBIDDEN)
    if leakage_features:
        raise ValueError(
            "Forbidden columns detected in model feature matrix:\n"
            + "\n".join(f"  - {c}" for c in leakage_features)
        )

    # Verify all feature columns exist.
    missing_features = sorted(set(feature_columns) - set(out.columns))
    if missing_features:
        raise ValueError(
            "Feature columns missing after engineering:\n"
            + "\n".join(f"  - {c}" for c in missing_features)
        )

    # Build compact reproducible output.
    features = out[feature_columns].copy()

    # Standardize categorical representation without one-hot encoding.
    # XGBoost/training stage will handle encoding consistently.
    for col in categorical_features:
        features[col] = features[col].astype("string").fillna("UNKNOWN")

    # Numeric conversion.
    for col in numeric_features:
        features[col] = pd.to_numeric(features[col], errors="coerce")

    # Restore targets after predictors.
    for target in TARGETS:
        features[target] = pd.to_numeric(target_values[target], errors="coerce")

    # Validate target values.
    for target in TARGETS:
        unique = set(features[target].dropna().unique())
        if not unique.issubset({0, 1}):
            raise ValueError(
                f"{target} contains values other than 0/1: {sorted(unique)}"
            )

    return features, numeric_features, categorical_features


def build_manifest(numeric_features, categorical_features):
    rows = []

    for feature in numeric_features:
        if feature in {
            "land_acquisition_ratio",
            "affected_family_per_parcel",
            "vulnerable_family_share",
            "compensation_pending_cr",
            "rnr_remaining_pct",
            "docs_remaining_pct",
        }:
            feature_type = "engineered_numeric"
        else:
            feature_type = "numeric"

        rows.append(
            {
                "feature": feature,
                "feature_type": feature_type,
                "model_role": "predictor",
                "availability": "snapshot_time",
                "notes": {
                    "land_acquisition_ratio": "land_to_acquire_ha / land_required_ha",
                    "affected_family_per_parcel": "affected_families / total_parcels",
                    "vulnerable_family_share": "vulnerable_families / affected_families",
                    "compensation_pending_cr": "compensation_awarded_cr - compensation_disbursed_cr",
                    "rnr_remaining_pct": "100 - rnr_progress_pct",
                    "docs_remaining_pct": "100 - docs_complete_pct",
                }.get(feature, ""),
            }
        )

    for feature in categorical_features:
        rows.append(
            {
                "feature": feature,
                "feature_type": "categorical",
                "model_role": "predictor",
                "availability": "snapshot_time",
                "notes": "",
            }
        )

    for target in TARGETS:
        rows.append(
            {
                "feature": target,
                "feature_type": "target",
                "model_role": "label",
                "availability": "future_window",
                "notes": "Forward-looking delay label; never used as predictor.",
            }
        )

    return pd.DataFrame(rows)


def main():
    print("=" * 72)
    print("BhoomiSetu ML Feature Builder")
    print("=" * 72)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}\n"
            "Run generate_synthetic_temporal_data.py first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)
    print(f"Input snapshots : {len(df)}")
    print(f"Input columns   : {len(df.columns)}")

    validate_input(df)

    features, numeric_features, categorical_features = build_features(df)
    manifest = build_manifest(numeric_features, categorical_features)

    # Missingness audit for final predictors.
    predictor_cols = numeric_features + categorical_features
    missing_counts = features[predictor_cols].isna().sum()
    missing_pct = missing_counts / len(features) * 100

    report_lines = [
        "BhoomiSetu ML Feature Build Report",
        "=" * 72,
        f"Input snapshots              : {len(df)}",
        f"Final predictor count        : {len(predictor_cols)}",
        f"  Numeric predictors         : {len(numeric_features)}",
        f"  Categorical predictors     : {len(categorical_features)}",
        f"Target count                 : {len(TARGETS)}",
        "",
        "Predictor leakage check      : PASS",
        f"Predictor missing values     : {int(missing_counts.sum())}",
        f"Maximum predictor missing %  : {missing_pct.max():.4f}%",
        "",
        "Excluded from predictors:",
    ]

    excluded = sorted(FORBIDDEN | {"event_count_so_far"})
    report_lines.extend(f"  - {c}" for c in excluded)

    report_lines.extend(
        [
            "",
            "Engineered features:",
            "  - land_acquisition_ratio",
            "  - affected_family_per_parcel",
            "  - vulnerable_family_share",
            "  - compensation_pending_cr",
            "  - rnr_remaining_pct",
            "  - docs_remaining_pct",
            "",
            "Important:",
            "This feature matrix is for prototype ML experimentation.",
            "It does not establish real-world Indian land-acquisition accuracy.",
        ]
    )

    if missing_counts.sum() > 0:
        report_lines.append(
            "\nWARNING: Predictor missingness exists; training pipeline must handle it."
        )
    else:
        report_lines.append("\nPredictor missingness check: PASS")

    output_features = OUTPUT_DIR / "ml_features.csv"
    output_manifest = OUTPUT_DIR / "feature_manifest.csv"
    output_report = OUTPUT_DIR / "feature_build_report.txt"

    features.to_csv(output_features, index=False)
    manifest.to_csv(output_manifest, index=False)
    output_report.write_text("\n".join(report_lines), encoding="utf-8")

    print()
    print("Final predictor count:", len(predictor_cols))
    print("  Numeric:", len(numeric_features))
    print("  Categorical:", len(categorical_features))
    print("Targets:", ", ".join(TARGETS))
    print("Predictor missing values:", int(missing_counts.sum()))
    print()
    print("Outputs:")
    print("  ", output_features)
    print("  ", output_manifest)
    print("  ", output_report)
    print()
    print("ML FEATURE BUILD: PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("ML FEATURE BUILD: FAIL")
        print(f"Reason: {exc}")
        sys.exit(1)
