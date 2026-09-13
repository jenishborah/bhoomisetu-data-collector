from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = (
    BASE_DIR
    / "output"
    / "normalized"
    / "bhoomirashi"
)

EVENT_FILE = (
    DATA_DIR
    / "temporal"
    / "observed_events.csv"
)

PROJECT_MASTER_FILE = (
    DATA_DIR
    / "project_master.csv"
)

OUTPUT_DIR = (
    DATA_DIR
    / "temporal"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "project_snapshots.csv"
)


# ============================================================
# Load input data
# ============================================================

if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"Observed event file not found: {EVENT_FILE}"
    )

if not PROJECT_MASTER_FILE.exists():
    raise FileNotFoundError(
        f"Project master file not found: "
        f"{PROJECT_MASTER_FILE}"
    )

events = pd.read_csv(EVENT_FILE)

master = pd.read_csv(
    PROJECT_MASTER_FILE
)

print("=" * 70)
print("BhoomiSetu Project Snapshot Builder")
print("=" * 70)

print()
print(
    f"Observed events : {len(events)}"
)

print(
    f"Projects        : "
    f"{events['project_id'].nunique()}"
)


# ============================================================
# Validate required columns
# ============================================================

required_event_columns = [
    "project_id",
    "event_date",
    "event_type",
    "notification_id",
    "notification_number",
    "status",
    "event_sequence",
    "days_since_previous_event",
    "days_since_first_event",
    "is_3a",
    "is_3A",
    "is_3D",
    "has_3a_by_date",
    "has_3A_by_date",
    "has_3D_by_date",
    "stage_as_of_event",
]

missing_event_columns = [
    column
    for column in required_event_columns
    if column not in events.columns
]

if missing_event_columns:
    raise ValueError(
        "Observed events are missing required columns: "
        + ", ".join(missing_event_columns)
    )


required_master_columns = [
    "project_id",
    "project_name",
    "project_number",
    "land_required_ha",
    "land_to_acquire_ha",
]

missing_master_columns = [
    column
    for column in required_master_columns
    if column not in master.columns
]

if missing_master_columns:
    raise ValueError(
        "Project master is missing required columns: "
        + ", ".join(missing_master_columns)
    )


# ============================================================
# Normalize dates
# ============================================================

events["event_date"] = pd.to_datetime(
    events["event_date"],
    errors="coerce",
)

if events["event_date"].isna().any():
    invalid = int(
        events["event_date"].isna().sum()
    )

    raise ValueError(
        f"{invalid} observed events have invalid dates."
    )


# ============================================================
# Sort events
# ============================================================

events = events.sort_values(
    [
        "project_id",
        "event_date",
        "notification_id",
    ]
).reset_index(drop=True)


# ============================================================
# Build snapshots
# ============================================================

snapshot_rows = []

for project_id, project_events in events.groupby(
    "project_id",
    sort=True,
):

    project_events = project_events.sort_values(
        [
            "event_date",
            "notification_id",
        ]
    ).reset_index(drop=True)

    project_start = project_events[
        "event_date"
    ].min()

    cumulative_3a = 0
    cumulative_3A = 0
    cumulative_3D = 0

    for index, row in project_events.iterrows():

        cumulative_3a += int(
            row["is_3a"]
        )

        cumulative_3A += int(
            row["is_3A"]
        )

        cumulative_3D += int(
            row["is_3D"]
        )

        snapshot_date = row[
            "event_date"
        ]

        days_since_start = (
            snapshot_date
            - project_start
        ).days

        # ----------------------------------------------------
        # Days since most recent event
        # ----------------------------------------------------

        if index == 0:

            days_since_previous = None

        else:

            previous_date = project_events.loc[
                index - 1,
                "event_date",
            ]

            days_since_previous = (
                snapshot_date
                - previous_date
            ).days

        # ----------------------------------------------------
        # Snapshot stage
        # ----------------------------------------------------

        stage = row[
            "stage_as_of_event"
        ]

        # ----------------------------------------------------
        # Snapshot row
        # ----------------------------------------------------

        snapshot_rows.append(
            {
                "project_id": project_id,

                "snapshot_id": (
                    f"{project_id}_"
                    f"{snapshot_date.strftime('%Y%m%d')}_"
                    f"{index + 1:03d}"
                ),

                "snapshot_date": (
                    snapshot_date.strftime(
                        "%Y-%m-%d"
                    )
                ),

                "current_stage": stage,

                "event_sequence": index + 1,

                "event_type": row[
                    "event_type"
                ],

                "notification_id": row[
                    "notification_id"
                ],

                "notification_number": row[
                    "notification_number"
                ],

                "status": row[
                    "status"
                ],

                # --------------------------------------------
                # Temporal state
                # --------------------------------------------

                "days_since_project_start":
                    days_since_start,

                "days_since_previous_event":
                    days_since_previous,

                "event_count_so_far":
                    index + 1,

                # --------------------------------------------
                # Cumulative stage state
                # --------------------------------------------

                "count_3a_so_far":
                    cumulative_3a,

                "count_3A_so_far":
                    cumulative_3A,

                "count_3D_so_far":
                    cumulative_3D,

                "has_3a":
                    int(cumulative_3a > 0),

                "has_3A":
                    int(cumulative_3A > 0),

                "has_3D":
                    int(cumulative_3D > 0),

                # --------------------------------------------
                # Project start / current stage
                # --------------------------------------------

                "project_start_date":
                    project_start.strftime(
                        "%Y-%m-%d"
                    ),

                "stage_entry_date":
                    snapshot_date.strftime(
                        "%Y-%m-%d"
                    ),
            }
        )


# ============================================================
# Convert to dataframe
# ============================================================

snapshots = pd.DataFrame(
    snapshot_rows
)

if snapshots.empty:
    raise ValueError(
        "No project snapshots were generated."
    )


# ============================================================
# Calculate days in current stage
# ============================================================

# For the first appearance of a stage, use the current
# event date as the stage entry date.
#
# For repeated events in the same stage, the stage itself
# does not necessarily restart. Therefore we determine the
# first date on which the current stage appeared.

snapshots["snapshot_date"] = pd.to_datetime(
    snapshots["snapshot_date"]
)

snapshots["project_start_date"] = pd.to_datetime(
    snapshots["project_start_date"]
)

snapshots["stage_entry_date"] = pd.to_datetime(
    snapshots["stage_entry_date"]
)

snapshots["stage_first_seen_date"] = (
    snapshots
    .groupby(
        [
            "project_id",
            "current_stage",
        ]
    )["snapshot_date"]
    .transform("min")
)

snapshots["days_in_current_stage"] = (
    snapshots["snapshot_date"]
    - snapshots["stage_first_seen_date"]
).dt.days


# ============================================================
# Attach static project information
# ============================================================

static_columns = [
    "project_id",
    "project_name",
    "project_number",
    "land_required_ha",
    "land_to_acquire_ha",
]

static = master[
    static_columns
].drop_duplicates(
    subset=["project_id"]
)

snapshots = snapshots.merge(
    static,
    on="project_id",
    how="left",
    validate="many_to_one",
)


# ============================================================
# Temporal safety flags
# ============================================================

# These flags explicitly document which fields are:
#
#   SAFE_OBSERVED
#   STATIC_REFERENCE
#
# We intentionally do not attach current acquisition,
# survey, party, or final-outcome fields to historical
# snapshots.

snapshots["project_id_data_class"] = (
    "STATIC_REFERENCE"
)

snapshots["project_name_data_class"] = (
    "STATIC_REFERENCE"
)

snapshots["project_number_data_class"] = (
    "STATIC_REFERENCE"
)

snapshots["land_required_ha_data_class"] = (
    "STATIC_REFERENCE"
)

snapshots["land_to_acquire_ha_data_class"] = (
    "STATIC_REFERENCE"
)

snapshots["temporal_state_data_class"] = (
    "OBSERVED_AS_OF_SNAPSHOT"
)


# ============================================================
# Reorder columns
# ============================================================

preferred_columns = [
    "snapshot_id",
    "project_id",
    "snapshot_date",
    "project_start_date",
    "current_stage",
    "stage_first_seen_date",
    "days_in_current_stage",
    "days_since_project_start",
    "days_since_previous_event",
    "event_sequence",
    "event_count_so_far",
    "event_type",
    "notification_id",
    "notification_number",
    "status",
    "count_3a_so_far",
    "count_3A_so_far",
    "count_3D_so_far",
    "has_3a",
    "has_3A",
    "has_3D",
    "project_name",
    "project_number",
    "land_required_ha",
    "land_to_acquire_ha",
    "project_id_data_class",
    "project_name_data_class",
    "project_number_data_class",
    "land_required_ha_data_class",
    "land_to_acquire_ha_data_class",
    "temporal_state_data_class",
]

remaining_columns = [
    column
    for column in snapshots.columns
    if column not in preferred_columns
]

snapshots = snapshots[
    preferred_columns
    + remaining_columns
]


# ============================================================
# Sort final dataset
# ============================================================

snapshots = snapshots.sort_values(
    [
        "project_id",
        "snapshot_date",
        "event_sequence",
    ]
).reset_index(drop=True)


# ============================================================
# Validation
# ============================================================

duplicate_snapshot_ids = int(
    snapshots["snapshot_id"]
    .duplicated()
    .sum()
)

if duplicate_snapshot_ids:

    raise ValueError(
        "Duplicate snapshot IDs detected: "
        f"{duplicate_snapshot_ids}"
    )


# ------------------------------------------------------------
# Validate stage chronology
# ------------------------------------------------------------

stage_order = {
    "3a": 1,
    "3A": 2,
    "3D": 3,
}

invalid_stage_transitions = []

for project_id, project_snapshots in (
    snapshots.groupby("project_id")
):

    previous_rank = 0

    for _, row in project_snapshots.iterrows():

        stage = row["current_stage"]

        rank = stage_order.get(
            stage,
            0,
        )

        # Repeated stages are allowed.
        # Moving backward is flagged for review.
        if rank < previous_rank:

            invalid_stage_transitions.append(
                {
                    "project_id": project_id,
                    "snapshot_date": row[
                        "snapshot_date"
                    ],
                    "current_stage": stage,
                    "previous_stage_rank":
                        previous_rank,
                    "current_stage_rank":
                        rank,
                }
            )

        previous_rank = max(
            previous_rank,
            rank,
        )


invalid_stage_df = pd.DataFrame(
    invalid_stage_transitions
)


# ============================================================
# Save snapshots
# ============================================================

snapshots.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Save transition validation
# ============================================================

transition_file = (
    OUTPUT_DIR
    / "snapshot_stage_validation.csv"
)

if invalid_stage_df.empty:

    invalid_stage_df = pd.DataFrame(
        columns=[
            "project_id",
            "snapshot_date",
            "current_stage",
            "previous_stage_rank",
            "current_stage_rank",
        ]
    )

invalid_stage_df.to_csv(
    transition_file,
    index=False,
)


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 70)
print("SNAPSHOT SUMMARY")
print("=" * 70)

print(
    f"Projects represented : "
    f"{snapshots['project_id'].nunique()}"
)

print(
    f"Snapshots generated  : "
    f"{len(snapshots)}"
)

print(
    f"Snapshot columns     : "
    f"{len(snapshots.columns)}"
)

print()
print("Snapshots by project:")

for project_id, count in (
    snapshots
    .groupby("project_id")
    .size()
    .sort_index()
    .items()
):

    print(
        f"  {project_id}: {count}"
    )

print()
print("Stage distribution:")

for stage, count in (
    snapshots["current_stage"]
    .value_counts()
    .sort_index()
    .items()
):

    print(
        f"  {stage}: {count}"
    )

print()
print(
    "Backward stage transitions: "
    f"{len(invalid_stage_df)}"
)

print()
print("Output:")
print(f"  {OUTPUT_FILE}")

print()
print("=" * 70)
print(
    "Project snapshot dataset created successfully."
)
print("=" * 70)