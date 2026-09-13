from __future__ import annotations

"""
BhoomiSetu ML-Readiness Statistical Audit

Purpose
-------
Audit the synthetic temporal dataset BEFORE model training.

This script checks:
1. Required files / schema
2. Missing values
3. Feature variance / constants
4. Numeric ranges and extreme outliers
5. Target prevalence
6. Numeric feature -> target signal
7. Categorical feature -> target signal
8. Feature correlation / redundancy
9. Project-level leakage risk
10. Temporal split readiness
11. Scenario separability (validation-only; scenario is NOT an ML feature)
12. Final PASS / WARN / FAIL recommendation

IMPORTANT
---------
This is an ML-readiness audit, not evidence of real-world model accuracy.
Synthetic relationships are generator assumptions, not empirical Indian
land-acquisition relationships.
"""

from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    BASE_DIR
    / "output"
    / "synthetic"
)

SNAPSHOTS_FILE = (
    INPUT_DIR
    / "synthetic_snapshots.csv"
)

OUTCOMES_FILE = (
    INPUT_DIR
    / "synthetic_project_outcomes.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "synthetic"
    / "audit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE = (
    OUTPUT_DIR
    / "ml_readiness_audit_report.txt"
)

FEATURE_REPORT_FILE = (
    OUTPUT_DIR
    / "feature_signal_report.csv"
)

CORRELATION_FILE = (
    OUTPUT_DIR
    / "feature_correlation_matrix.csv"
)

TARGET_SUMMARY_FILE = (
    OUTPUT_DIR
    / "target_summary.csv"
)

SCENARIO_SUMMARY_FILE = (
    OUTPUT_DIR
    / "scenario_signal_summary.csv"
)


# ============================================================
# Dataset schema
# ============================================================

REQUIRED_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "project_id",
    "snapshot_date",
    "project_start_date",
    "current_stage",
    "stage_entry_date",
    "days_in_current_stage",
    "days_since_project_start",
    "event_count_so_far",
    "land_required_ha",
    "land_to_acquire_ha",
    "total_parcels",
    "affected_families",
    "vulnerable_families",
    "pending_approvals_count",
    "approval_dept",
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
    "terrain_type",
    "urban_rural",
    "forest_involved",
    "proximity_km_to_urban",
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
    "is_censored",
]

REQUIRED_OUTCOME_COLUMNS = [
    "project_id",
    "scenario_ground_truth",
    "total_snapshots",
    "observation_start_date",
    "observation_end_date",
    "final_stage",
    "delayed",
    "delay_days",
    "first_delay_day",
    "delay_30_event_count",
    "delay_60_event_count",
    "delay_90_event_count",
    "is_censored",
]


TARGETS = [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]


NUMERIC_FEATURES = [
    "days_in_current_stage",
    "days_since_project_start",
    "event_count_so_far",
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


CATEGORICAL_FEATURES = [
    "current_stage",
    "approval_dept",
    "terrain_type",
    "urban_rural",
]


FORBIDDEN_FEATURES = [
    "snapshot_id",
    "project_id",
    "snapshot_date",
    "project_start_date",
    "stage_entry_date",
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
    "is_censored",
    "scenario",
    "scenario_ground_truth",
    "first_delay_day",
    "delay_days",
    "delay_30_event_count",
    "delay_60_event_count",
    "delay_90_event_count",
    "delayed",
]


# ============================================================
# Thresholds
# ============================================================

MISSING_WARN_PCT = 1.0
MISSING_FAIL_PCT = 5.0

CONSTANT_WARN_COUNT = 1

HIGH_CORRELATION_THRESHOLD = 0.90

MIN_CATEGORY_COUNT = 10

SIGNAL_MEAN_DIFF_WARN = 0.10

MIN_PROJECTS_FOR_TEMPORAL_EVAL = 100


# ============================================================
# Helpers
# ============================================================

issues = []
warnings_list = []
passes = []


def record_pass(message):
    passes.append(message)


def record_warning(message):
    warnings_list.append(message)


def record_issue(message):
    issues.append(message)


def pct(value, total):
    if total == 0:
        return 0.0
    return float(value) / float(total) * 100.0


def safe_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def auc_like_mean_separation(feature, target):
    """
    For binary target, estimate directional separation using the
    probability that a randomly selected positive has a larger
    feature value than a randomly selected negative.

    This is equivalent to ROC-AUC for a single numeric feature,
    computed without sklearn.
    """
    frame = pd.DataFrame(
        {
            "x": pd.to_numeric(
                feature,
                errors="coerce",
            ),
            "y": pd.to_numeric(
                target,
                errors="coerce",
            ),
        }
    ).dropna()

    if frame.empty:
        return np.nan

    positives = frame.loc[
        frame["y"] == 1,
        "x",
    ]

    negatives = frame.loc[
        frame["y"] == 0,
        "x",
    ]

    if len(positives) == 0 or len(negatives) == 0:
        return np.nan

    # Mann-Whitney formulation with average ranks.
    combined = frame["x"].rank(
        method="average"
    )

    rank_sum_positive = combined.loc[
        frame["y"] == 1
    ].sum()

    n_pos = len(positives)
    n_neg = len(negatives)

    u = (
        rank_sum_positive
        - n_pos * (n_pos + 1) / 2
    )

    return float(
        u / (n_pos * n_neg)
    )


def entropy(values):
    counts = (
        pd.Series(values)
        .value_counts(
            normalize=True,
            dropna=False,
        )
    )

    return float(
        -sum(
            p * math.log2(p)
            for p in counts
            if p > 0
        )
    )


def categorical_signal(frame, feature, target):
    """
    Returns category-level target prevalence and max prevalence
    separation. This is descriptive only.
    """
    data = frame[
        [feature, target]
    ].copy()

    data[target] = pd.to_numeric(
        data[target],
        errors="coerce",
    )

    data = data.dropna(
        subset=[target]
    )

    if data.empty:
        return np.nan, np.nan, 0

    grouped = (
        data.groupby(
            feature,
            dropna=False,
        )[target]
        .agg(
            count="size",
            positive_rate="mean",
        )
        .reset_index()
    )

    if len(grouped) <= 1:
        return 0.0, 0.0, len(grouped)

    overall = data[target].mean()

    grouped["abs_diff"] = (
        grouped["positive_rate"]
        - overall
    ).abs()

    return (
        float(grouped["abs_diff"].max()),
        float(grouped["positive_rate"].max()
              - grouped["positive_rate"].min()),
        len(grouped),
    )


# ============================================================
# Load
# ============================================================

print("=" * 78)
print("BhoomiSetu ML-Readiness Statistical Audit")
print("=" * 78)

print()

if not SNAPSHOTS_FILE.exists():
    raise FileNotFoundError(
        f"Missing snapshots file: {SNAPSHOTS_FILE}"
    )

if not OUTCOMES_FILE.exists():
    raise FileNotFoundError(
        f"Missing outcomes file: {OUTCOMES_FILE}"
    )

snapshots = pd.read_csv(
    SNAPSHOTS_FILE
)

outcomes = pd.read_csv(
    OUTCOMES_FILE
)

print(
    f"Snapshots : {len(snapshots)}"
)

print(
    f"Outcomes  : {len(outcomes)}"
)

print(
    f"Projects  : {snapshots['project_id'].nunique()}"
)


# ============================================================
# 1. Schema
# ============================================================

print()
print("=" * 78)
print("1. SCHEMA CHECK")
print("=" * 78)

missing_snapshot_columns = [
    c
    for c in REQUIRED_SNAPSHOT_COLUMNS
    if c not in snapshots.columns
]

missing_outcome_columns = [
    c
    for c in REQUIRED_OUTCOME_COLUMNS
    if c not in outcomes.columns
]

if missing_snapshot_columns:
    record_issue(
        "Missing snapshot columns: "
        + ", ".join(
            missing_snapshot_columns
        )
    )
else:
    record_pass(
        "All required snapshot columns present."
    )

if missing_outcome_columns:
    record_issue(
        "Missing outcome columns: "
        + ", ".join(
            missing_outcome_columns
        )
    )
else:
    record_pass(
        "All required outcome columns present."
    )

if issues:
    print("Schema: FAIL")
    for item in issues:
        print(f"  FAIL: {item}")
    raise ValueError(
        "Schema requirements failed."
    )

print("Schema: PASS")


# ============================================================
# 2. Missingness
# ============================================================

print()
print("=" * 78)
print("2. MISSINGNESS CHECK")
print("=" * 78)

feature_columns = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

missing_rows = []

for feature in feature_columns:
    missing_count = int(
        snapshots[feature]
        .isna()
        .sum()
    )

    missing_pct = pct(
        missing_count,
        len(snapshots),
    )

    missing_rows.append(
        {
            "feature": feature,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
        }
    )

    if missing_pct >= MISSING_FAIL_PCT:
        record_issue(
            f"{feature}: "
            f"{missing_pct:.2f}% missing"
        )

    elif missing_pct >= MISSING_WARN_PCT:
        record_warning(
            f"{feature}: "
            f"{missing_pct:.2f}% missing"
        )

    else:
        record_pass(
            f"{feature}: missingness acceptable"
        )

missing_df = pd.DataFrame(
    missing_rows
)

print(
    missing_df.to_string(
        index=False
    )
)

if any(
    item.startswith(
        tuple(
            f"{f}:" for f in NUMERIC_FEATURES
        )
    )
    for item in []
):
    pass

if any(
    row["missing_pct"]
    >= MISSING_FAIL_PCT
    for row in missing_rows
):
    print(
        "Missingness: FAIL"
    )
else:
    print(
        "Missingness: PASS/WARN"
    )


# ============================================================
# 3. Numeric feature health
# ============================================================

print()
print("=" * 78)
print("3. NUMERIC FEATURE HEALTH")
print("=" * 78)

feature_health = []

for feature in NUMERIC_FEATURES:

    x = pd.to_numeric(
        snapshots[feature],
        errors="coerce",
    )

    non_null = x.dropna()

    if non_null.empty:
        record_issue(
            f"{feature}: no numeric observations"
        )
        continue

    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        outlier_count = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(
            (
                (non_null < lower)
                | (non_null > upper)
            ).sum()
        )

    unique_count = int(
        non_null.nunique()
    )

    std = float(
        non_null.std()
    )

    feature_health.append(
        {
            "feature": feature,
            "unique_count": unique_count,
            "mean": float(non_null.mean()),
            "std": std,
            "min": float(non_null.min()),
            "q1": float(q1),
            "median": float(non_null.median()),
            "q3": float(q3),
            "max": float(non_null.max()),
            "iqr_outlier_count": outlier_count,
            "iqr_outlier_pct": pct(
                outlier_count,
                len(non_null),
            ),
        }
    )

    if unique_count <= 1:
        record_issue(
            f"{feature}: constant feature"
        )

    elif unique_count <= 3:
        record_warning(
            f"{feature}: very low cardinality "
            f"({unique_count} unique values)"
        )

health_df = pd.DataFrame(
    feature_health
)

print(
    health_df.to_string(
        index=False
    )
)

constant_features = health_df.loc[
    health_df["unique_count"] <= 1,
    "feature",
].tolist()

if constant_features:
    print(
        "Numeric feature health: FAIL"
    )
else:
    print(
        "Numeric feature health: PASS/WARN"
    )


# ============================================================
# 4. Target distribution
# ============================================================

print()
print("=" * 78)
print("4. TARGET DISTRIBUTION")
print("=" * 78)

target_rows = []

for target in TARGETS:

    y = pd.to_numeric(
        snapshots[target],
        errors="coerce",
    )

    positive = int(
        (y == 1).sum()
    )

    negative = int(
        (y == 0).sum()
    )

    missing = int(
        y.isna().sum()
    )

    positive_pct = pct(
        positive,
        len(y),
    )

    target_rows.append(
        {
            "target": target,
            "positive": positive,
            "negative": negative,
            "missing": missing,
            "positive_pct": positive_pct,
        }
    )

    print(
        f"{target:<22}"
        f"positive={positive:>5} "
        f"negative={negative:>5} "
        f"positive_pct={positive_pct:>6.2f}%"
    )

    if positive == 0 or negative == 0:
        record_issue(
            f"{target}: only one class present"
        )

    elif positive_pct < 1:
        record_warning(
            f"{target}: very rare positive class "
            f"({positive_pct:.2f}%)"
        )

    else:
        record_pass(
            f"{target}: both classes present"
        )

target_df = pd.DataFrame(
    target_rows
)

target_df.to_csv(
    TARGET_SUMMARY_FILE,
    index=False,
)

print(
    "Target distribution: PASS/WARN"
)


# ============================================================
# 5. Numeric feature -> target signal
# ============================================================

print()
print("=" * 78)
print("5. NUMERIC FEATURE / TARGET SIGNAL")
print("=" * 78)

signal_rows = []

for target in TARGETS:

    y = snapshots[target]

    for feature in NUMERIC_FEATURES:

        x = pd.to_numeric(
            snapshots[feature],
            errors="coerce",
        )

        valid = (
            x.notna()
            & y.notna()
        )

        if valid.sum() < 20:
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        spearman = safe_float(
            x_valid.corr(
                y_valid,
                method="spearman",
            )
        )

        auc = auc_like_mean_separation(
            x_valid,
            y_valid,
        )

        positive_mean = safe_float(
            x_valid[
                y_valid == 1
            ].mean()
        )

        negative_mean = safe_float(
            x_valid[
                y_valid == 0
            ].mean()
        )

        if (
            pd.notna(positive_mean)
            and pd.notna(negative_mean)
            and negative_mean != 0
        ):
            relative_difference = (
                positive_mean
                - negative_mean
            ) / abs(
                negative_mean
            )
        else:
            relative_difference = np.nan

        signal_rows.append(
            {
                "target": target,
                "feature": feature,
                "spearman": spearman,
                "single_feature_auc": auc,
                "positive_mean": positive_mean,
                "negative_mean": negative_mean,
                "relative_mean_difference":
                    relative_difference,
                "n": int(valid.sum()),
            }
        )

signal_df = pd.DataFrame(
    signal_rows
)

signal_df.to_csv(
    FEATURE_REPORT_FILE,
    index=False,
)

for target in TARGETS:

    subset = signal_df[
        signal_df["target"] == target
    ].copy()

    subset["abs_auc_distance"] = (
        subset["single_feature_auc"]
        - 0.5
    ).abs()

    top = (
        subset
        .sort_values(
            "abs_auc_distance",
            ascending=False,
        )
        .head(8)
    )

    print()
    print(
        f"Top single-feature signals for {target}:"
    )

    for _, row in top.iterrows():

        print(
            f"  {row['feature']:<30}"
            f"AUC={row['single_feature_auc']:.3f} "
            f"Spearman={row['spearman']:.3f}"
        )

print()
print(
    "Numeric signal audit: COMPLETE"
)


# ============================================================
# 6. Categorical feature signal
# ============================================================

print()
print("=" * 78)
print("6. CATEGORICAL FEATURE / TARGET SIGNAL")
print("=" * 78)

categorical_rows = []

for target in TARGETS:

    for feature in CATEGORICAL_FEATURES:

        max_abs_diff, spread, category_count = (
            categorical_signal(
                snapshots,
                feature,
                target,
            )
        )

        counts = (
            snapshots[feature]
            .value_counts(
                dropna=False
            )
        )

        rare_categories = int(
            (
                counts
                < MIN_CATEGORY_COUNT
            ).sum()
        )

        categorical_rows.append(
            {
                "target": target,
                "feature": feature,
                "category_count":
                    category_count,
                "max_abs_prevalence_difference":
                    max_abs_diff,
                "prevalence_spread":
                    spread,
                "rare_category_count":
                    rare_categories,
            }
        )

        print(
            f"{target:<22}"
            f"{feature:<24}"
            f"categories={category_count:<3} "
            f"spread={spread:.4f} "
            f"rare={rare_categories}"
        )

        if rare_categories > 0:
            record_warning(
                f"{feature}: "
                f"{rare_categories} rare categories"
            )

categorical_df = pd.DataFrame(
    categorical_rows
)

print(
    "Categorical signal audit: COMPLETE"
)


# ============================================================
# 7. Correlation / redundancy
# ============================================================

print()
print("=" * 78)
print("7. FEATURE CORRELATION / REDUNDANCY")
print("=" * 78)

numeric_frame = snapshots[
    NUMERIC_FEATURES
].apply(
    pd.to_numeric,
    errors="coerce",
)

correlation = numeric_frame.corr(
    method="spearman"
)

correlation.to_csv(
    CORRELATION_FILE
)

high_pairs = []

for i, feature_a in enumerate(
    correlation.columns
):

    for feature_b in (
        correlation.columns[
            i + 1:
        :]
    ):

        value = correlation.loc[
            feature_a,
            feature_b,
        ]

        if (
            pd.notna(value)
            and abs(value)
            >= HIGH_CORRELATION_THRESHOLD
        ):

            high_pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "spearman": float(value),
                }
            )

            record_warning(
                "High correlation: "
                f"{feature_a} ↔ {feature_b} "
                f"({value:.3f})"
            )

if high_pairs:

    high_corr_df = pd.DataFrame(
        high_pairs
    )

    print(
        high_corr_df.to_string(
            index=False
        )
    )

    print(
        f"High-correlation pairs: "
        f"{len(high_pairs)}"
    )

else:

    print(
        "No feature pairs exceed "
        f"|Spearman| >= {HIGH_CORRELATION_THRESHOLD:.2f}"
    )

record_pass(
    "Correlation matrix generated."
)


# ============================================================
# 8. Project-level leakage / repeated snapshots
# ============================================================

print()
print("=" * 78)
print("8. PROJECT-LEVEL LEAKAGE RISK")
print("=" * 78)

project_count = (
    snapshots["project_id"]
    .nunique()
)

snapshot_project_counts = (
    snapshots
    .groupby("project_id")
    .size()
)

multi_snapshot_projects = int(
    (
        snapshot_project_counts > 1
    ).sum()
)

max_snapshots = int(
    snapshot_project_counts.max()
)

print(
    f"Unique projects             : "
    f"{project_count}"
)

print(
    f"Projects with >1 snapshot   : "
    f"{multi_snapshot_projects}"
)

print(
    f"Maximum snapshots/project   : "
    f"{max_snapshots}"
)

if multi_snapshot_projects > 0:

    record_warning(
        "Repeated snapshots exist. "
        "Random row-level train/test splitting "
        "would create project leakage."
    )

    print(
        "Leakage risk: WARNING"
    )

else:

    record_pass(
        "No repeated projects."
    )

    print(
        "Leakage risk: PASS"
    )


# ============================================================
# 9. Temporal split readiness
# ============================================================

print()
print("=" * 78)
print("9. TEMPORAL SPLIT READINESS")
print("=" * 78)

snapshots["_snapshot_date"] = pd.to_datetime(
    snapshots["snapshot_date"],
    errors="coerce",
)

project_dates = (
    snapshots
    .groupby("project_id")[
        "_snapshot_date"
    ]
    .agg(
        first_date="min",
        last_date="max",
    )
    .reset_index()
)

print(
    f"Earliest snapshot: "
    f"{project_dates['first_date'].min().date()}"
)

print(
    f"Latest snapshot  : "
    f"{project_dates['last_date'].max().date()}"
)

if project_count >= MIN_PROJECTS_FOR_TEMPORAL_EVAL:

    # Project-level chronological split.
    project_dates = (
        project_dates
        .sort_values(
            "first_date"
        )
        .reset_index(
            drop=True
        )
    )

    n = len(project_dates)

    train_end = int(
        n * 0.70
    )

    valid_end = int(
        n * 0.85
    )

    train_projects = set(
        project_dates.iloc[
            :train_end
        ]["project_id"]
    )

    valid_projects = set(
        project_dates.iloc[
            train_end:valid_end
        ]["project_id"]
    )

    test_projects = set(
        project_dates.iloc[
            valid_end:
        ]["project_id"]
    )

    overlap_tv = (
        train_projects
        & valid_projects
    )

    overlap_tt = (
        train_projects
        & test_projects
    )

    overlap_vt = (
        valid_projects
        & test_projects
    )

    print(
        f"Proposed train projects: "
        f"{len(train_projects)}"
    )

    print(
        f"Proposed valid projects: "
        f"{len(valid_projects)}"
    )

    print(
        f"Proposed test projects : "
        f"{len(test_projects)}"
    )

    if (
        overlap_tv
        or overlap_tt
        or overlap_vt
    ):

        record_issue(
            "Project overlap detected "
            "in proposed temporal split."
        )

        print(
            "Temporal split readiness: FAIL"
        )

    else:

        record_pass(
            "Project-disjoint chronological split "
            "is feasible."
        )

        print(
            "Temporal split readiness: PASS"
        )

else:

    record_warning(
        "Too few projects for robust temporal "
        "holdout assessment."
    )

    print(
        "Temporal split readiness: WARNING"
    )


# ============================================================
# 10. Snapshot label/event timing sanity
# ============================================================

print()
print("=" * 78)
print("10. LABEL TIMING SANITY")
print("=" * 78)

timing_rows = []

for target in TARGETS:

    positive_by_day = (
        snapshots.loc[
            snapshots[target] == 1,
            "days_since_project_start",
        ]
    )

    if positive_by_day.empty:
        timing_rows.append(
            {
                "target": target,
                "positive_count": 0,
                "median_positive_project_day":
                    np.nan,
                "min_positive_project_day":
                    np.nan,
                "max_positive_project_day":
                    np.nan,
            }
        )

        continue

    timing_rows.append(
        {
            "target": target,
            "positive_count":
                int(len(positive_by_day)),
            "median_positive_project_day":
                float(
                    positive_by_day.median()
                ),
            "min_positive_project_day":
                float(
                    positive_by_day.min()
                ),
            "max_positive_project_day":
                float(
                    positive_by_day.max()
                ),
        }
    )

timing_df = pd.DataFrame(
    timing_rows
)

print(
    timing_df.to_string(
        index=False
    )
)

print(
    "Label timing sanity: COMPLETE"
)


# ============================================================
# 11. Scenario separability (validation only)
# ============================================================

print()
print("=" * 78)
print("11. SCENARIO SEPARABILITY — VALIDATION ONLY")
print("=" * 78)

if "scenario_ground_truth" not in outcomes.columns:

    record_warning(
        "scenario_ground_truth unavailable."
    )

else:

    scenario_map = outcomes[
        [
            "project_id",
            "scenario_ground_truth",
        ]
    ].drop_duplicates(
        "project_id"
    )

    scenario_frame = snapshots.merge(
        scenario_map,
        on="project_id",
        how="left",
        validate="many_to_one",
    )

    scenario_rows = []

    selected_features = [
        "pending_approvals_count",
        "file_pending_days",
        "docs_complete_pct",
        "court_cases_count",
        "grievances_30d",
        "rnr_progress_pct",
        "rnr_consent_pct",
        "disbursal_pct",
    ]

    for scenario_name, group in (
        scenario_frame
        .groupby(
            "scenario_ground_truth"
        )
    ):

        row = {
            "scenario":
                scenario_name,
            "snapshot_count":
                len(group),
        }

        for feature in selected_features:

            row[
                f"{feature}_mean"
            ] = float(
                pd.to_numeric(
                    group[feature],
                    errors="coerce",
                ).mean()
            )

        scenario_rows.append(
            row
        )

    scenario_df = pd.DataFrame(
        scenario_rows
    ).sort_values(
        "scenario"
    )

    scenario_df.to_csv(
        SCENARIO_SUMMARY_FILE,
        index=False,
    )

    print(
        scenario_df.to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT: scenario_ground_truth is "
        "for validation only and must NOT be used "
        "as an ML feature."
    )

    record_pass(
        "Scenario separability summary generated "
        "without using scenario as a model feature."
    )


# ============================================================
# 12. Forbidden-feature audit
# ============================================================

print()
print("=" * 78)
print("12. FORBIDDEN / LEAKAGE FEATURE AUDIT")
print("=" * 78)

present_forbidden = [
    c
    for c in FORBIDDEN_FEATURES
    if c in snapshots.columns
]

# IDs/date fields are expected to exist in the raw snapshot
# table. The purpose is to prevent them from accidentally
# entering the model feature matrix.

print(
    "Columns intentionally excluded from model matrix:"
)

for feature in present_forbidden:
    print(
        f"  - {feature}"
    )

record_pass(
    "Raw snapshot table contains expected "
    "identifier/date/target fields that must "
    "be excluded during model training."
)


# ============================================================
# 13. Recommended modeling feature matrix
# ============================================================

print()
print("=" * 78)
print("13. MODEL FEATURE MATRIX DESIGN")
print("=" * 78)

model_features = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

unexpected_model_features = [
    f
    for f in model_features
    if f in FORBIDDEN_FEATURES
]

missing_model_features = [
    f
    for f in model_features
    if f not in snapshots.columns
]

print(
    f"Candidate numeric features     : "
    f"{len(NUMERIC_FEATURES)}"
)

print(
    f"Candidate categorical features  : "
    f"{len(CATEGORICAL_FEATURES)}"
)

print(
    f"Candidate total model features  : "
    f"{len(model_features)}"
)

if unexpected_model_features:
    record_issue(
        "Forbidden columns included in model "
        "feature list: "
        + ", ".join(
            unexpected_model_features
        )
    )

if missing_model_features:
    record_issue(
        "Candidate model features missing: "
        + ", ".join(
            missing_model_features
        )
    )

if (
    not unexpected_model_features
    and not missing_model_features
):

    record_pass(
        "Candidate model matrix contains no "
        "explicit future-target fields."
    )

    print(
        "Model feature design: PASS"
    )

else:

    print(
        "Model feature design: FAIL"
    )


# ============================================================
# 14. Final readiness decision
# ============================================================

print()
print("=" * 78)
print("FINAL ML-READINESS DECISION")
print("=" * 78)

print(
    f"PASS checks    : {len(passes)}"
)

print(
    f"WARN checks    : {len(warnings_list)}"
)

print(
    f"FAIL checks    : {len(issues)}"
)

if issues:

    status = "FAIL"

elif warnings_list:

    status = "PASS_WITH_WARNINGS"

else:

    status = "PASS"

print()
print(
    f"STATUS: {status}"
)

print()

print(
    "Interpretation:"
)

if status == "FAIL":

    print(
        "The dataset requires correction before "
        "baseline model training."
    )

elif status == "PASS_WITH_WARNINGS":

    print(
        "The dataset is structurally suitable for "
        "prototype ML experimentation, but the "
        "warnings must be documented and addressed "
        "before treating model metrics as robust."
    )

else:

    print(
        "The dataset passes the statistical "
        "ML-readiness audit for prototype "
        "experimentation."
    )

print()
print(
    "This audit does NOT establish real-world "
    "accuracy, causal relationships, statutory "
    "deadlines, or Indian land-acquisition delay rates."
)


# ============================================================
# Write human-readable report
# ============================================================

with REPORT_FILE.open(
    "w",
    encoding="utf-8",
) as report:

    report.write(
        "BhoomiSetu ML-Readiness Statistical Audit\n"
    )

    report.write(
        "=" * 78
        + "\n\n"
    )

    report.write(
        f"Snapshots : {len(snapshots)}\n"
    )

    report.write(
        f"Projects  : "
        f"{snapshots['project_id'].nunique()}\n"
    )

    report.write(
        f"Outcomes  : {len(outcomes)}\n\n"
    )

    report.write(
        f"STATUS: {status}\n\n"
    )

    report.write(
        "FAILURES\n"
        + "-" * 40
        + "\n"
    )

    if issues:
        for item in issues:
            report.write(
                f"- {item}\n"
            )
    else:
        report.write(
            "None\n"
        )

    report.write(
        "\nWARNINGS\n"
        + "-" * 40
        + "\n"
    )

    if warnings_list:
        for item in warnings_list:
            report.write(
                f"- {item}\n"
            )
    else:
        report.write(
            "None\n"
        )

    report.write(
        "\nPASS CHECKS\n"
        + "-" * 40
        + "\n"
    )

    for item in passes:
        report.write(
            f"- {item}\n"
        )

    report.write(
        "\nOUTPUT FILES\n"
        + "-" * 40
        + "\n"
    )

    report.write(
        f"- {REPORT_FILE}\n"
    )

    report.write(
        f"- {FEATURE_REPORT_FILE}\n"
    )

    report.write(
        f"- {CORRELATION_FILE}\n"
    )

    report.write(
        f"- {TARGET_SUMMARY_FILE}\n"
    )

    report.write(
        f"- {SCENARIO_SUMMARY_FILE}\n"
    )

    report.write(
        "\nIMPORTANT\n"
        "Synthetic data is for prototype development only. "
        "It must not be presented as evidence of real-world "
        "Indian land-acquisition delay rates or model accuracy.\n"
    )


# ============================================================
# Cleanup
# ============================================================

if "_snapshot_date" in snapshots.columns:
    snapshots.drop(
        columns=["_snapshot_date"],
        inplace=True,
    )


print()
print("=" * 78)
print("AUDIT OUTPUTS")
print("=" * 78)

print(
    f"Report       : {REPORT_FILE}"
)

print(
    f"Feature      : {FEATURE_REPORT_FILE}"
)

print(
    f"Correlation  : {CORRELATION_FILE}"
)

print(
    f"Targets      : {TARGET_SUMMARY_FILE}"
)

print(
    f"Scenarios    : {SCENARIO_SUMMARY_FILE}"
)

print()
print("=" * 78)
print(
    f"ML-READINESS AUDIT COMPLETE — {status}"
)
print("=" * 78)

if status == "FAIL":
    raise ValueError(
        "ML-readiness audit failed. "
        "Review the generated report before model training."
    )
