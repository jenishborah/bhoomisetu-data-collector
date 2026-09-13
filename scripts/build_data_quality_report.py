from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "output" / "normalized" / "bhoomirashi"

PROJECT_MASTER = DATA_DIR / "project_master.csv"
PROJECT_FEATURES = DATA_DIR / "project_features.csv"
PROJECT_TIMELINE = DATA_DIR / "project_timeline.csv"
NOTIFICATION_TIMELINE = DATA_DIR / "notification_timeline.csv"

OUTPUT_DIR = DATA_DIR / "quality"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Helpers
# ============================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV and fail clearly if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")

    return pd.read_csv(path)


def safe_int(value: Any) -> int:
    """Convert a value to int safely."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def dataframe_summary(name: str, df: pd.DataFrame) -> dict:
    """Return basic dataframe statistics."""
    total_cells = df.shape[0] * df.shape[1]

    missing_cells = int(df.isna().sum().sum())

    duplicate_rows = int(df.duplicated().sum())

    return {
        "dataset": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "total_cells": int(total_cells),
        "missing_cells": missing_cells,
        "missing_pct": (
            round((missing_cells / total_cells) * 100, 4)
            if total_cells
            else 0.0
        ),
        "duplicate_rows": duplicate_rows,
    }


def column_quality(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Generate column-level quality statistics."""

    rows = []

    for column in df.columns:
        series = df[column]

        missing = int(series.isna().sum())
        non_missing = int(series.notna().sum())
        unique = int(series.nunique(dropna=True))

        # Percentage of non-null values
        completeness = (
            (non_missing / len(df)) * 100
            if len(df)
            else 0.0
        )

        # Percentage of dominant value
        dominant_pct = 0.0

        if non_missing > 0:
            value_counts = series.value_counts(dropna=True)
            dominant_pct = (
                value_counts.iloc[0] / non_missing
            ) * 100

        rows.append(
            {
                "dataset": dataset_name,
                "column": column,
                "dtype": str(series.dtype),
                "rows": len(df),
                "missing_count": missing,
                "missing_pct": round(
                    (missing / len(df)) * 100, 4
                ) if len(df) else 0.0,
                "unique_values": unique,
                "completeness_pct": round(
                    completeness, 4
                ),
                "dominant_value_pct": round(
                    dominant_pct, 4
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Load datasets
# ============================================================

print("=" * 70)
print("BhoomiSetu Data Quality Report")
print("=" * 70)

master = load_csv(PROJECT_MASTER)
features = load_csv(PROJECT_FEATURES)
timeline = load_csv(PROJECT_TIMELINE)
notification_timeline = load_csv(NOTIFICATION_TIMELINE)

print()
print("Loaded datasets:")
print(f"  Project master       : {len(master):>6}")
print(f"  Project features     : {len(features):>6}")
print(f"  Project timeline     : {len(timeline):>6}")
print(f"  Notification timeline: {len(notification_timeline):>6}")


# ============================================================
# Dataset summaries
# ============================================================

summaries = [
    dataframe_summary("project_master", master),
    dataframe_summary("project_features", features),
    dataframe_summary("project_timeline", timeline),
    dataframe_summary(
        "notification_timeline",
        notification_timeline,
    ),
]

summary_df = pd.DataFrame(summaries)

summary_df.to_csv(
    OUTPUT_DIR / "dataset_summary.csv",
    index=False,
)


# ============================================================
# Column-level quality
# ============================================================

quality_frames = [
    column_quality(master, "project_master"),
    column_quality(features, "project_features"),
    column_quality(timeline, "project_timeline"),
    column_quality(
        notification_timeline,
        "notification_timeline",
    ),
]

column_quality_df = pd.concat(
    quality_frames,
    ignore_index=True,
)

column_quality_df.to_csv(
    OUTPUT_DIR / "column_quality.csv",
    index=False,
)


# ============================================================
# Project-level checks
# ============================================================

project_checks = []

if "project_id" in master.columns:

    project_ids = master["project_id"]

    project_checks.append(
        {
            "check": "project_count",
            "value": int(project_ids.nunique()),
            "status": "INFO",
            "description": "Unique projects in project master",
        }
    )

    duplicate_projects = int(
        project_ids.duplicated().sum()
    )

    project_checks.append(
        {
            "check": "duplicate_project_ids",
            "value": duplicate_projects,
            "status": (
                "PASS"
                if duplicate_projects == 0
                else "WARNING"
            ),
            "description": (
                "Project IDs should be unique "
                "in project master"
            ),
        }
    )


# ============================================================
# Stage coverage
# ============================================================

if "latest_stage" in master.columns:

    stage_counts = (
        master["latest_stage"]
        .fillna("UNKNOWN")
        .value_counts()
    )

    for stage, count in stage_counts.items():

        project_checks.append(
            {
                "check": f"latest_stage_{stage}",
                "value": int(count),
                "status": "INFO",
                "description": (
                    "Projects whose latest known "
                    "stage is this stage"
                ),
            }
        )


# ============================================================
# 3D coverage
# ============================================================

if "survey_count" in master.columns:

    no_3d = int(
        (master["survey_count"].fillna(0) == 0).sum()
    )

    has_3d = int(
        (master["survey_count"].fillna(0) > 0).sum()
    )

    project_checks.extend(
        [
            {
                "check": "projects_with_3d_detail",
                "value": has_3d,
                "status": "INFO",
                "description": (
                    "Projects with extracted 3D survey records"
                ),
            },
            {
                "check": "projects_without_3d_detail",
                "value": no_3d,
                "status": "INFO",
                "description": (
                    "Projects without extracted 3D survey records"
                ),
            },
        ]
    )


# ============================================================
# Notification coverage
# ============================================================

if "notification_count" in master.columns:

    notification_counts = master[
        "notification_count"
    ].fillna(0)

    projects_without_notifications = int(
        (notification_counts == 0).sum()
    )

    project_checks.append(
        {
            "check": "projects_without_notifications",
            "value": projects_without_notifications,
            "status": (
                "PASS"
                if projects_without_notifications == 0
                else "INFO"
            ),
            "description": (
                "Projects without notification records"
            ),
        }
    )


# ============================================================
# Acquisition progress checks
# ============================================================

if "acquisition_completion_pct" in master.columns:

    completion = pd.to_numeric(
        master["acquisition_completion_pct"],
        errors="coerce",
    )

    negative_completion = int(
        (completion < 0).sum()
    )

    above_100 = int(
        (completion > 100).sum()
    )

    project_checks.extend(
        [
            {
                "check": "negative_acquisition_completion",
                "value": negative_completion,
                "status": (
                    "PASS"
                    if negative_completion == 0
                    else "WARNING"
                ),
                "description": (
                    "Acquisition completion below 0%"
                ),
            },
            {
                "check": "acquisition_completion_above_100",
                "value": above_100,
                "status": (
                    "PASS"
                    if above_100 == 0
                    else "WARNING"
                ),
                "description": (
                    "Acquisition completion above 100%"
                ),
            },
        ]
    )


# ============================================================
# Timeline consistency
# ============================================================

if "first_3a_date" in master.columns:
    if "first_3A_date" in master.columns:

        first_3a = pd.to_datetime(
            master["first_3a_date"],
            errors="coerce",
        )

        first_3A = pd.to_datetime(
            master["first_3A_date"],
            errors="coerce",
        )

        invalid_order = int(
            (
                first_3A.notna()
                & first_3a.notna()
                & (first_3A < first_3a)
            ).sum()
        )

        project_checks.append(
            {
                "check": "3A_before_3a",
                "value": invalid_order,
                "status": (
                    "PASS"
                    if invalid_order == 0
                    else "WARNING"
                ),
                "description": (
                    "3A notification occurring before "
                    "3a notification"
                ),
            }
        )


# ============================================================
# Known test/demo project detection
# ============================================================

test_projects = []

name_columns = [
    column
    for column in [
        "project_name",
        "project_number",
    ]
    if column in master.columns
]

if name_columns:

    mask = pd.Series(
        False,
        index=master.index,
    )

    for column in name_columns:

        mask = mask | (
            master[column]
            .fillna("")
            .astype(str)
            .str.contains(
                "test",
                case=False,
                na=False,
            )
        )

    test_projects = master.loc[
        mask,
        [
            column
            for column in [
                "project_id",
                "project_name",
                "project_number",
            ]
            if column in master.columns
        ],
    ].copy()

if len(test_projects) > 0:

    test_projects.to_csv(
        OUTPUT_DIR / "detected_test_projects.csv",
        index=False,
    )

    project_checks.append(
        {
            "check": "possible_test_projects",
            "value": int(len(test_projects)),
            "status": "WARNING",
            "description": (
                "Projects whose name/number contains "
                "'test'; review before ML training"
            ),
        }
    )
else:

    project_checks.append(
        {
            "check": "possible_test_projects",
            "value": 0,
            "status": "PASS",
            "description": (
                "No obvious test projects detected "
                "by name/number"
            ),
        }
    )


# ============================================================
# Feature variation checks
# ============================================================

feature_variation = []

for column in features.columns:

    if column == "project_id":
        continue

    series = features[column]

    unique_count = series.nunique(
        dropna=True
    )

    if unique_count <= 1:

        feature_variation.append(
            {
                "column": column,
                "unique_values": int(unique_count),
                "issue": "NO_VARIATION",
                "recommendation": (
                    "Do not use as predictive feature "
                    "until more varied data exists."
                ),
            }
        )

    elif unique_count <= 2:

        feature_variation.append(
            {
                "column": column,
                "unique_values": int(unique_count),
                "issue": "LOW_VARIATION",
                "recommendation": (
                    "Review before using as an ML feature."
                ),
            }
        )

feature_variation_df = pd.DataFrame(
    feature_variation
)

feature_variation_df.to_csv(
    OUTPUT_DIR / "low_variation_features.csv",
    index=False,
)


# ============================================================
# Potential leakage fields
# ============================================================

leakage_keywords = [
    "actual",
    "delay_days",
    "delayed",
    "outcome",
    "failed",
    "completion_date",
    "future",
]

leakage_candidates = []

for column in features.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in leakage_keywords
    ):

        leakage_candidates.append(
            {
                "column": column,
                "reason": (
                    "Column name suggests possible "
                    "future/outcome information."
                ),
                "action": (
                    "Review before using for prediction. "
                    "Only information available at snapshot "
                    "time may be used as an input feature."
                ),
            }
        )

leakage_df = pd.DataFrame(
    leakage_candidates
)

leakage_df.to_csv(
    OUTPUT_DIR / "potential_leakage_features.csv",
    index=False,
)


# ============================================================
# Overall readiness assessment
# ============================================================

warnings = []

for check in project_checks:

    if check["status"] == "WARNING":
        warnings.append(check)


if len(master) < 50:

    warnings.append(
        {
            "check": "small_project_count",
            "value": len(master),
            "status": "WARNING",
            "description": (
                "Current project count is too small "
                "for meaningful production ML training."
            ),
        }
    )


if len(features) < 50:

    warnings.append(
        {
            "check": "small_feature_dataset",
            "value": len(features),
            "status": "WARNING",
            "description": (
                "Current feature dataset contains "
                "too few project observations."
            ),
        }
    )


if warnings:

    overall_status = "NOT_READY_FOR_PRODUCTION_ML"

else:

    overall_status = "READY_FOR_NEXT_PIPELINE_STAGE"


# ============================================================
# Save project checks
# ============================================================

project_checks_df = pd.DataFrame(project_checks)

project_checks_df.to_csv(
    OUTPUT_DIR / "project_checks.csv",
    index=False,
)


# ============================================================
# JSON summary
# ============================================================

report = {
    "dataset": "BhoomiSetu BhoomiRashi prototype",
    "overall_status": overall_status,
    "project_count": int(len(master)),
    "feature_rows": int(len(features)),
    "feature_columns": int(len(features.columns)),
    "timeline_rows": int(len(timeline)),
    "notification_timeline_rows": int(
        len(notification_timeline)
    ),
    "warnings": warnings,
    "outputs": {
        "dataset_summary": str(
            OUTPUT_DIR / "dataset_summary.csv"
        ),
        "column_quality": str(
            OUTPUT_DIR / "column_quality.csv"
        ),
        "project_checks": str(
            OUTPUT_DIR / "project_checks.csv"
        ),
        "low_variation_features": str(
            OUTPUT_DIR / "low_variation_features.csv"
        ),
        "potential_leakage_features": str(
            OUTPUT_DIR / "potential_leakage_features.csv"
        ),
        "detected_test_projects": str(
            OUTPUT_DIR / "detected_test_projects.csv"
        ),
    },
}


with open(
    OUTPUT_DIR / "quality_report.json",
    "w",
    encoding="utf-8",
) as file:

    json.dump(
        report,
        file,
        indent=2,
        default=str,
    )


# ============================================================
# Console output
# ============================================================

print()
print("=" * 70)
print("QUALITY SUMMARY")
print("=" * 70)

print(f"Projects                 : {len(master)}")
print(f"Project features         : {len(features)}")
print(f"Feature columns          : {len(features.columns)}")
print(f"Project timeline records : {len(timeline)}")
print(
    f"Notification events      : "
    f"{len(notification_timeline)}"
)

print()
print(
    f"Overall status            : "
    f"{overall_status}"
)

print()
print("Warnings:")

if warnings:

    for warning in warnings:

        print(
            f"  - {warning['check']}: "
            f"{warning['description']}"
        )

else:

    print("  None")

print()
print("Reports written to:")
print(f"  {OUTPUT_DIR}")

print()
print("Done.")