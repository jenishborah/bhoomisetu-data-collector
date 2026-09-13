from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "output"
    / "synthetic"
)

PROJECTS_FILE = (
    DATA_DIR
    / "synthetic_projects.csv"
)

SNAPSHOTS_FILE = (
    DATA_DIR
    / "synthetic_snapshots.csv"
)

OUTCOMES_FILE = (
    DATA_DIR
    / "synthetic_project_outcomes.csv"
)


# ============================================================
# Load
# ============================================================

projects = pd.read_csv(
    PROJECTS_FILE
)

snapshots = pd.read_csv(
    SNAPSHOTS_FILE
)

outcomes = pd.read_csv(
    OUTCOMES_FILE
)


print("=" * 70)
print(
    "BhoomiSetu Synthetic Data Statistical Validation"
)
print("=" * 70)

print()

print(
    f"Projects : {len(projects)}"
)

print(
    f"Snapshots: {len(snapshots)}"
)

print(
    f"Outcomes : {len(outcomes)}"
)


# ============================================================
# 1. Required columns
# ============================================================

print()
print("=" * 70)
print("1. REQUIRED COLUMN CHECK")
print("=" * 70)


required_project_columns = [
    "project_id",
    "project_type",
    "implementing_agency",
    "state",
    "district",
    "scenario",
    "land_required_ha",
    "land_to_acquire_ha",
    "total_parcels",
    "affected_families",
    "vulnerable_families",
    "date_initiation",
]


required_snapshot_columns = [
    "snapshot_id",
    "project_id",
    "snapshot_date",
    "project_start_date",
    "current_stage",
    "stage_entry_date",
    "days_in_current_stage",
    "days_since_project_start",
    "pending_approvals_count",
    "file_pending_days",
    "docs_complete_pct",
    "court_cases_count",
    "case_age_months",
    "title_disputes_parcels",
    "grievances_30d",
    "disbursal_pct",
    "rnr_progress_pct",
    "rnr_consent_pct",
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]


required_outcome_columns = [
    "project_id",
    "scenario_ground_truth",
    "delayed",
    "delay_days",
    "first_delay_day",
    "is_censored",
]


def check_columns(
    dataframe,
    required,
    name,
):

    missing = [
        column
        for column in required
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"{name} missing columns: "
            + ", ".join(missing)
        )

    print(
        f"{name}: PASS"
    )


check_columns(
    projects,
    required_project_columns,
    "Projects",
)

check_columns(
    snapshots,
    required_snapshot_columns,
    "Snapshots",
)

check_columns(
    outcomes,
    required_outcome_columns,
    "Outcomes",
)


# ============================================================
# 2. Numeric range checks
# ============================================================

print()
print("=" * 70)
print("2. FEATURE RANGE CHECK")
print("=" * 70)


range_rules = {
    "docs_complete_pct": (
        0,
        100,
    ),

    "disbursal_pct": (
        0,
        100,
    ),

    "rnr_progress_pct": (
        0,
        100,
    ),

    "rnr_consent_pct": (
        0,
        100,
    ),

    "pending_approvals_count": (
        0,
        np.inf,
    ),

    "file_pending_days": (
        0,
        np.inf,
    ),

    "court_cases_count": (
        0,
        np.inf,
    ),

    "case_age_months": (
        0,
        np.inf,
    ),

    "title_disputes_parcels": (
        0,
        np.inf,
    ),

    "grievances_30d": (
        0,
        np.inf,
    ),

    "days_in_current_stage": (
        0,
        np.inf,
    ),

    "days_since_project_start": (
        0,
        np.inf,
    ),
}


range_failures = 0


for column, (
    lower,
    upper,
) in range_rules.items():

    values = pd.to_numeric(
        snapshots[column],
        errors="coerce",
    )

    invalid = (
        values.isna()
        | (values < lower)
        | (values > upper)
    )

    count = int(
        invalid.sum()
    )

    print(
        f"  {column:<32} "
        f"invalid={count}"
    )

    range_failures += count


if range_failures > 0:

    raise ValueError(
        "Feature range validation failed."
    )

print()
print(
    "Feature ranges: PASS"
)


# ============================================================
# 3. Temporal ordering
# ============================================================

print()
print("=" * 70)
print("3. TEMPORAL ORDERING CHECK")
print("=" * 70)


snapshots[
    "snapshot_date"
] = pd.to_datetime(
    snapshots[
        "snapshot_date"
    ]
)

snapshots[
    "project_start_date"
] = pd.to_datetime(
    snapshots[
        "project_start_date"
    ]
)

snapshots = snapshots.sort_values(
    [
        "project_id",
        "snapshot_date",
    ]
)


temporal_failures = 0


for project_id, group in (
    snapshots.groupby(
        "project_id"
    )
):

    dates = group[
        "snapshot_date"
    ].tolist()

    if dates != sorted(dates):

        temporal_failures += 1

    if any(
        dates[i] >= dates[i + 1]
        for i in range(
            len(dates) - 1
        )
    ):

        temporal_failures += 1


print(
    f"Projects with temporal ordering "
    f"issues: {temporal_failures}"
)


if temporal_failures > 0:

    raise ValueError(
        "Temporal ordering failed."
    )


print(
    "Temporal ordering: PASS"
)


# ============================================================
# 4. Snapshot interval consistency
# ============================================================

print()
print("=" * 70)
print("4. SNAPSHOT INTERVAL CHECK")
print("=" * 70)


interval_issues = 0


for project_id, group in (
    snapshots.groupby(
        "project_id"
    )
):

    group = group.sort_values(
        "snapshot_date"
    )

    dates = group[
        "snapshot_date"
    ].tolist()

    for i in range(
        len(dates) - 1
    ):

        difference = (
            dates[i + 1]
            - dates[i]
        ).days

        if (
            difference
            != 30
        ):

            interval_issues += 1


print(
    f"Non-30-day snapshot intervals: "
    f"{interval_issues}"
)


if interval_issues > 0:

    raise ValueError(
        "Unexpected snapshot interval detected."
    )


print(
    "Snapshot interval consistency: PASS"
)


# ============================================================
# 5. Project-level snapshot counts
# ============================================================

print()
print("=" * 70)
print("5. SNAPSHOT COUNT CHECK")
print("=" * 70)


snapshot_counts = (
    snapshots
    .groupby(
        "project_id"
    )
    .size()
)


invalid_counts = (
    (
        snapshot_counts
        < 8
    )
    |
    (
        snapshot_counts
        > 15
    )
).sum()


print(
    f"Projects outside 8–15 snapshots: "
    f"{invalid_counts}"
)


if invalid_counts > 0:

    raise ValueError(
        "Snapshot count validation failed."
    )


print(
    "Snapshot counts: PASS"
)


# ============================================================
# 6. Forward label monotonicity
# ============================================================

print()
print("=" * 70)
print("6. FORWARD LABEL CHECK")
print("=" * 70)


invalid_30_60 = (
    (
        snapshots[
            "delay_next_30d"
        ] == 1
    )
    &
    (
        snapshots[
            "delay_next_60d"
        ] == 0
    )
).sum()


invalid_60_90 = (
    (
        snapshots[
            "delay_next_60d"
        ] == 1
    )
    &
    (
        snapshots[
            "delay_next_90d"
        ] == 0
    )
).sum()


print(
    f"30d=1 and 60d=0: "
    f"{invalid_30_60}"
)

print(
    f"60d=1 and 90d=0: "
    f"{invalid_60_90}"
)


if (
    invalid_30_60 > 0
    or invalid_60_90 > 0
):

    raise ValueError(
        "Forward labels are not monotonic."
    )


print(
    "Forward label monotonicity: PASS"
)


# ============================================================
# 7. Label distribution
# ============================================================

print()
print("=" * 70)
print("7. LABEL DISTRIBUTION")
print("=" * 70)


for column in [
    "delay_next_30d",
    "delay_next_60d",
    "delay_next_90d",
]:

    positive = int(
        snapshots[
            column
        ].sum()
    )

    total = len(
        snapshots
    )

    negative = (
        total
        - positive
    )

    percentage = (
        positive
        / total
        * 100
    )

    print(
        f"{column}: "
        f"positive={positive}, "
        f"negative={negative}, "
        f"positive_pct={percentage:.2f}%"
    )


# ============================================================
# 8. Scenario → feature relationships
# ============================================================

print()
print("=" * 70)
print(
    "8. SCENARIO / OBSERVABLE FEATURE CHECK"
)
print("=" * 70)


scenario_features = [
    "pending_approvals_count",
    "file_pending_days",
    "docs_complete_pct",
    "court_cases_count",
    "case_age_months",
    "title_disputes_parcels",
    "grievances_30d",
    "disbursal_pct",
    "rnr_progress_pct",
    "rnr_consent_pct",
]


# Join hidden scenario to snapshots ONLY for validation.
# This joined dataframe must never be used as the ML dataset.

validation = snapshots.merge(
    projects[
        [
            "project_id",
            "scenario",
        ]
    ],
    on="project_id",
    how="left",
    validate="many_to_one",
)


if validation[
    "scenario"
].isna().any():

    raise ValueError(
        "Some snapshots could not be linked "
        "to a project scenario."
    )


scenario_means = (
    validation
    .groupby(
        "scenario"
    )[scenario_features]
    .mean()
)


print()
print(
    "Selected scenario means:"
)

selected_features = [
    "pending_approvals_count",
    "file_pending_days",
    "docs_complete_pct",
    "court_cases_count",
    "grievances_30d",
    "rnr_progress_pct",
]


print(
    scenario_means[
        selected_features
    ].round(2).to_string()
)


# ============================================================
# 9. Scenario separation sanity checks
# ============================================================

print()
print(
    "Scenario-specific sanity checks:"
)


def mean_for(
    scenario,
    feature,
):

    return float(
        scenario_means.loc[
            scenario,
            feature,
        ]
    )


checks = []


# Administrative scenario should have more pending approvals
checks.append(
    (
        "ADMIN approvals > NORMAL",
        mean_for(
            "ADMINISTRATIVE_BOTTLENECK",
            "pending_approvals_count",
        )
        >
        mean_for(
            "NORMAL",
            "pending_approvals_count",
        ),
    )
)


# Legal scenario should have more court cases
checks.append(
    (
        "LEGAL court cases > NORMAL",
        mean_for(
            "LEGAL_DELAY",
            "court_cases_count",
        )
        >
        mean_for(
            "NORMAL",
            "court_cases_count",
        ),
    )
)


# Documentation scenario should have lower documentation
checks.append(
    (
        "DOCUMENTATION docs < NORMAL",
        mean_for(
            "DOCUMENTATION_DELAY",
            "docs_complete_pct",
        )
        <
        mean_for(
            "NORMAL",
            "docs_complete_pct",
        ),
    )
)


# R&R scenario should have lower R&R progress
checks.append(
    (
        "RNR progress < NORMAL",
        mean_for(
            "RNR_DELAY",
            "rnr_progress_pct",
        )
        <
        mean_for(
            "NORMAL",
            "rnr_progress_pct",
        ),
    )
)


for description, result in checks:

    print(
        f"  {description}: "
        f"{'PASS' if result else 'FAIL'}"
    )


failed_checks = [
    description
    for description, result in checks
    if not result
]


if failed_checks:

    raise ValueError(
        "Scenario relationship checks failed: "
        + "; ".join(
            failed_checks
        )
    )


print(
    "Scenario relationship checks: PASS"
)


# ============================================================
# 10. Outcome consistency
# ============================================================

print()
print("=" * 70)
print(
    "9. OUTCOME CONSISTENCY"
)
print("=" * 70)


delayed_without_day = (
    (
        outcomes[
            "delayed"
        ] == 1
    )
    &
    (
        outcomes[
            "first_delay_day"
        ].isna()
    )
).sum()


not_delayed_with_day = (
    (
        outcomes[
            "delayed"
        ] == 0
    )
    &
    (
        outcomes[
            "first_delay_day"
        ].notna()
    )
).sum()


print(
    f"Delayed without first_delay_day: "
    f"{delayed_without_day}"
)

print(
    f"Not-delayed with first_delay_day: "
    f"{not_delayed_with_day}"
)


if (
    delayed_without_day > 0
    or not_delayed_with_day > 0
):

    raise ValueError(
        "Outcome consistency failed."
    )


print(
    "Outcome consistency: PASS"
)


# ============================================================
# 11. Duplicate checks
# ============================================================

print()
print("=" * 70)
print(
    "10. DUPLICATE CHECK"
)
print("=" * 70)


duplicate_projects = (
    projects[
        "project_id"
    ]
    .duplicated()
    .sum()
)

duplicate_snapshots = (
    snapshots[
        "snapshot_id"
    ]
    .duplicated()
    .sum()
)

duplicate_outcomes = (
    outcomes[
        "project_id"
    ]
    .duplicated()
    .sum()
)


print(
    f"Duplicate project IDs   : "
    f"{duplicate_projects}"
)

print(
    f"Duplicate snapshot IDs  : "
    f"{duplicate_snapshots}"
)

print(
    f"Duplicate outcome IDs   : "
    f"{duplicate_outcomes}"
)


if (
    duplicate_projects > 0
    or duplicate_snapshots > 0
    or duplicate_outcomes > 0
):

    raise ValueError(
        "Duplicate identifiers detected."
    )


print(
    "Duplicate check: PASS"
)


# ============================================================
# Final status
# ============================================================

print()
print("=" * 70)
print(
    "SYNTHETIC DATA VALIDATION RESULT"
)
print("=" * 70)

print(
    "STATUS: PASS"
)

print()
print(
    "The synthetic dataset is internally "
    "consistent and suitable for prototype "
    "ML pipeline development."
)

print()
print(
    "IMPORTANT:"
)

print(
    "This does NOT establish real-world "
    "model accuracy or Indian delay rates."
)

print(
    "Real-world validation requires sufficiently "
    "large historical acquisition data."
)

print("=" * 70)