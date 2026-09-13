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
    / "stage_cycles.csv"
)


# ============================================================
# Load
# ============================================================

if not EVENT_FILE.exists():
    raise FileNotFoundError(
        f"Observed event file not found: {EVENT_FILE}"
    )

events = pd.read_csv(EVENT_FILE)

print("=" * 70)
print("BhoomiSetu Stage Cycle Builder")
print("=" * 70)

print()
print(f"Observed events : {len(events)}")
print(
    f"Projects        : "
    f"{events['project_id'].nunique()}"
)


# ============================================================
# Validate
# ============================================================

required_columns = [
    "project_id",
    "event_date",
    "event_type",
    "notification_id",
    "notification_number",
    "status",
    "event_sequence",
]

missing = [
    column
    for column in required_columns
    if column not in events.columns
]

if missing:
    raise ValueError(
        "Observed events are missing required columns: "
        + ", ".join(missing)
    )


# ============================================================
# Normalize
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
        f"{invalid} events have invalid event dates."
    )


events["event_type"] = (
    events["event_type"]
    .fillna("")
    .astype(str)
    .str.strip()
)


events = events.sort_values(
    [
        "project_id",
        "event_date",
        "notification_id",
    ]
).reset_index(drop=True)


# ============================================================
# Stage cycle logic
# ============================================================

cycle_rows = []

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

    cycle_number = 0

    cycle_start_date = None

    cycle_completed = False

    cycle_event_sequence = 0

    last_stage = None

    for _, event in project_events.iterrows():

        event_type = event["event_type"]

        # ----------------------------------------------------
        # Start a new cycle
        #
        # A new cycle begins when:
        #
        #   event = 3a
        #   AND the previous cycle has already reached 3D
        #
        # For the first 3a, cycle 1 is created.
        # ----------------------------------------------------

        if event_type == "3a":

            if cycle_number == 0:

                cycle_number = 1

                cycle_start_date = (
                    event["event_date"]
                )

                cycle_completed = False

                cycle_event_sequence = 0

            elif cycle_completed:

                cycle_number += 1

                cycle_start_date = (
                    event["event_date"]
                )

                cycle_completed = False

                cycle_event_sequence = 0

            # If the current cycle is still open,
            # another 3a is treated as part of that
            # same cycle.

        # ----------------------------------------------------
        # Handle unexpected events before first 3a
        # ----------------------------------------------------

        if cycle_number == 0:

            cycle_number = 1

            cycle_start_date = (
                event["event_date"]
            )

            cycle_completed = False

            cycle_event_sequence = 0

        # ----------------------------------------------------
        # Increment cycle event sequence
        # ----------------------------------------------------

        cycle_event_sequence += 1

        # ----------------------------------------------------
        # Mark cycle completion at 3D
        # ----------------------------------------------------

        if event_type == "3D":

            cycle_completed = True

        # ----------------------------------------------------
        # Calculate elapsed time
        # ----------------------------------------------------

        cycle_days_elapsed = (
            event["event_date"]
            - cycle_start_date
        ).days

        # ----------------------------------------------------
        # Cycle stage
        # ----------------------------------------------------

        if event_type == "3a":

            cycle_stage = "3a"

        elif event_type == "3A":

            cycle_stage = "3A"

        elif event_type == "3D":

            cycle_stage = "3D"

        else:

            cycle_stage = event_type

        # ----------------------------------------------------
        # Cycle status
        # ----------------------------------------------------

        if cycle_completed:

            cycle_status = "COMPLETED"

        else:

            cycle_status = "OPEN"

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        cycle_rows.append(
            {
                "project_id":
                    project_id,

                "cycle_number":
                    cycle_number,

                "cycle_id":
                    (
                        f"{project_id}_"
                        f"C{cycle_number:02d}"
                    ),

                "cycle_start_date":
                    cycle_start_date.strftime(
                        "%Y-%m-%d"
                    ),

                "event_date":
                    event["event_date"].strftime(
                        "%Y-%m-%d"
                    ),

                "cycle_event_sequence":
                    cycle_event_sequence,

                "project_event_sequence":
                    int(event["event_sequence"]),

                "event_type":
                    event_type,

                "cycle_stage":
                    cycle_stage,

                "notification_id":
                    event["notification_id"],

                "notification_number":
                    event["notification_number"],

                "status":
                    event["status"],

                "cycle_days_elapsed":
                    cycle_days_elapsed,

                "cycle_status":
                    cycle_status,

                "cycle_completed":
                    int(cycle_completed),
            }
        )

        last_stage = cycle_stage


# ============================================================
# DataFrame
# ============================================================

cycles = pd.DataFrame(
    cycle_rows
)

if cycles.empty:
    raise ValueError(
        "No stage-cycle records were generated."
    )


# ============================================================
# Sort
# ============================================================

cycles = cycles.sort_values(
    [
        "project_id",
        "cycle_number",
        "event_date",
        "project_event_sequence",
    ]
).reset_index(drop=True)


# ============================================================
# Cycle summary columns
# ============================================================

cycle_summary = (
    cycles
    .groupby(
        [
            "project_id",
            "cycle_number",
            "cycle_id",
        ],
        as_index=False,
    )
    .agg(
        cycle_start_date=(
            "cycle_start_date",
            "first",
        ),

        cycle_last_event_date=(
            "event_date",
            "last",
        ),

        cycle_event_count=(
            "cycle_event_sequence",
            "max",
        ),

        has_3a=(
            "event_type",
            lambda x: int(
                (x == "3a").any()
            ),
        ),

        has_3A=(
            "event_type",
            lambda x: int(
                (x == "3A").any()
            ),
        ),

        has_3D=(
            "event_type",
            lambda x: int(
                (x == "3D").any()
            ),
        ),

        cycle_status=(
            "cycle_status",
            "last",
        ),

        cycle_completed=(
            "cycle_completed",
            "max",
        ),
    )
)


# ============================================================
# Cycle duration
# ============================================================

cycle_summary["cycle_start_date"] = (
    pd.to_datetime(
        cycle_summary[
            "cycle_start_date"
        ]
    )
)

cycle_summary["cycle_last_event_date"] = (
    pd.to_datetime(
        cycle_summary[
            "cycle_last_event_date"
        ]
    )
)

cycle_summary["cycle_duration_days"] = (
    cycle_summary[
        "cycle_last_event_date"
    ]
    - cycle_summary[
        "cycle_start_date"
    ]
).dt.days


# ============================================================
# Latest stage within cycle
# ============================================================

latest_stage = (
    cycles
    .sort_values(
        [
            "project_id",
            "cycle_number",
            "event_date",
            "project_event_sequence",
        ]
    )
    .groupby(
        [
            "project_id",
            "cycle_number",
        ],
        as_index=False,
    )
    .tail(1)
    [
        [
            "project_id",
            "cycle_number",
            "cycle_stage",
        ]
    ]
    .rename(
        columns={
            "cycle_stage":
                "latest_cycle_stage"
        }
    )
)


cycle_summary = cycle_summary.merge(
    latest_stage,
    on=[
        "project_id",
        "cycle_number",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# Reorder summary
# ============================================================

summary_columns = [
    "project_id",
    "cycle_number",
    "cycle_id",
    "cycle_start_date",
    "cycle_last_event_date",
    "cycle_duration_days",
    "cycle_event_count",
    "latest_cycle_stage",
    "cycle_status",
    "cycle_completed",
    "has_3a",
    "has_3A",
    "has_3D",
]

cycle_summary = cycle_summary[
    summary_columns
]


# ============================================================
# Save detailed event-level cycle dataset
# ============================================================

cycles.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Save cycle summary
# ============================================================

SUMMARY_FILE = (
    OUTPUT_DIR
    / "stage_cycle_summary.csv"
)

cycle_summary.to_csv(
    SUMMARY_FILE,
    index=False,
)


# ============================================================
# Validation
# ============================================================

validation_errors = []

for project_id, project_cycles in (
    cycle_summary.groupby("project_id")
):

    project_cycles = (
        project_cycles
        .sort_values("cycle_number")
        .reset_index(drop=True)
    )

    expected_cycle = 1

    for _, cycle in project_cycles.iterrows():

        actual_cycle = int(
            cycle["cycle_number"]
        )

        if actual_cycle != expected_cycle:

            validation_errors.append(
                {
                    "project_id":
                        project_id,
                    "expected_cycle":
                        expected_cycle,
                    "actual_cycle":
                        actual_cycle,
                }
            )

        expected_cycle += 1


# ============================================================
# Validation output
# ============================================================

VALIDATION_FILE = (
    OUTPUT_DIR
    / "stage_cycle_validation.csv"
)

validation_df = pd.DataFrame(
    validation_errors
)

if validation_df.empty:

    validation_df = pd.DataFrame(
        columns=[
            "project_id",
            "expected_cycle",
            "actual_cycle",
        ]
    )

validation_df.to_csv(
    VALIDATION_FILE,
    index=False,
)


# ============================================================
# Console summary
# ============================================================

print()
print("=" * 70)
print("STAGE CYCLE SUMMARY")
print("=" * 70)

print(
    f"Projects represented : "
    f"{cycle_summary['project_id'].nunique()}"
)

print(
    f"Cycles generated     : "
    f"{len(cycle_summary)}"
)

print(
    f"Event records        : "
    f"{len(cycles)}"
)

print()

print("Cycles by project:")

for project_id, count in (
    cycle_summary
    .groupby("project_id")
    .size()
    .sort_index()
    .items()
):

    print(
        f"  {project_id}: {count}"
    )


print()

print("Cycle summary:")

display_columns = [
    "project_id",
    "cycle_number",
    "cycle_start_date",
    "cycle_last_event_date",
    "cycle_duration_days",
    "cycle_event_count",
    "latest_cycle_stage",
    "cycle_status",
]

print(
    cycle_summary[
        display_columns
    ].to_string(index=False)
)


print()

print(
    "Validation errors: "
    f"{len(validation_df)}"
)

print()

print("Outputs:")

print(
    f"  Event-level cycles : "
    f"{OUTPUT_FILE}"
)

print(
    f"  Cycle summary      : "
    f"{SUMMARY_FILE}"
)

print(
    f"  Validation         : "
    f"{VALIDATION_FILE}"
)

print()

if validation_df.empty:

    print(
        "Stage-cycle validation: PASS"
    )

else:

    print(
        "Stage-cycle validation: WARNING"
    )

print()

print("=" * 70)
print(
    "Stage cycle dataset created successfully."
)
print("=" * 70)