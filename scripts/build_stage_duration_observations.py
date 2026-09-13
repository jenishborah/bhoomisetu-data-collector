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

TEMPORAL_DIR = DATA_DIR / "temporal"

INPUT_FILE = (
    TEMPORAL_DIR
    / "stage_cycles.csv"
)

OUTPUT_FILE = (
    TEMPORAL_DIR
    / "stage_duration_observations.csv"
)


# ============================================================
# Load
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

events = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("BhoomiSetu Stage Duration Observation Builder")
print("=" * 70)

print()
print(f"Input event records : {len(events)}")
print(
    f"Projects            : "
    f"{events['project_id'].nunique()}"
)


# ============================================================
# Validate
# ============================================================

required_columns = [
    "project_id",
    "cycle_number",
    "cycle_id",
    "event_date",
    "event_type",
    "notification_id",
    "notification_number",
    "status",
    "cycle_event_sequence",
]

missing = [
    column
    for column in required_columns
    if column not in events.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
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

    raise ValueError(
        "One or more event_date values are invalid."
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
        "cycle_number",
        "event_date",
        "cycle_event_sequence",
    ]
).reset_index(drop=True)


# ============================================================
# Build transition observations
# ============================================================

observations = []


for (
    project_id,
    cycle_number,
), group in events.groupby(
    [
        "project_id",
        "cycle_number",
    ],
    sort=True,
):

    group = group.sort_values(
        [
            "event_date",
            "cycle_event_sequence",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # We deliberately inspect adjacent observed events.
    #
    # Only these transitions are currently benchmarkable:
    #
    #     3a → 3A
    #     3A → 3D
    #
    # Repeated 3a or repeated 3A/3D events are retained,
    # but they are not treated as separate transition
    # benchmarks unless they represent the requested pair.
    # --------------------------------------------------------

    for i in range(len(group) - 1):

        start = group.iloc[i]
        end = group.iloc[i + 1]

        from_stage = start["event_type"]
        to_stage = end["event_type"]

        # ----------------------------------------------------
        # Only accepted transitions
        # ----------------------------------------------------

        if (
            from_stage == "3a"
            and to_stage == "3A"
        ):

            transition = "3a_to_3A"

        elif (
            from_stage == "3A"
            and to_stage == "3D"
        ):

            transition = "3A_to_3D"

        else:

            continue

        # ----------------------------------------------------
        # Duration
        # ----------------------------------------------------

        duration_days = (
            end["event_date"]
            - start["event_date"]
        ).days

        if duration_days < 0:

            raise ValueError(
                "Negative transition duration found "
                f"for project {project_id}, "
                f"cycle {cycle_number}."
            )

        # ----------------------------------------------------
        # Observation quality
        # ----------------------------------------------------

        observation_quality = "OBSERVED"

        # ----------------------------------------------------
        # Append
        # ----------------------------------------------------

        observations.append(
            {
                "project_id":
                    project_id,

                "cycle_number":
                    cycle_number,

                "cycle_id":
                    start["cycle_id"],

                "transition":
                    transition,

                "from_stage":
                    from_stage,

                "to_stage":
                    to_stage,

                "start_event_date":
                    start["event_date"].strftime(
                        "%Y-%m-%d"
                    ),

                "end_event_date":
                    end["event_date"].strftime(
                        "%Y-%m-%d"
                    ),

                "duration_days":
                    duration_days,

                "start_notification_id":
                    start["notification_id"],

                "end_notification_id":
                    end["notification_id"],

                "start_notification_number":
                    start["notification_number"],

                "end_notification_number":
                    end["notification_number"],

                "start_status":
                    start["status"],

                "end_status":
                    end["status"],

                "start_event_sequence":
                    int(
                        start[
                            "cycle_event_sequence"
                        ]
                    ),

                "end_event_sequence":
                    int(
                        end[
                            "cycle_event_sequence"
                        ]
                    ),

                "observation_quality":
                    observation_quality,

                "usable_for_empirical_benchmark":
                    1,
            }
        )


# ============================================================
# Create DataFrame
# ============================================================

observations_df = pd.DataFrame(
    observations
)


if observations_df.empty:

    raise ValueError(
        "No valid stage-duration observations were generated."
    )


# ============================================================
# Sort
# ============================================================

observations_df = (
    observations_df
    .sort_values(
        [
            "project_id",
            "cycle_number",
            "start_event_date",
            "start_event_sequence",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# Save
# ============================================================

observations_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 70)
print("OBSERVATION SUMMARY")
print("=" * 70)

print(
    f"Transition observations : "
    f"{len(observations_df)}"
)

print(
    f"Projects represented    : "
    f"{observations_df['project_id'].nunique()}"
)

print()

print("Observations by transition:")

transition_counts = (
    observations_df
    .groupby("transition")
    .size()
)

for transition, count in (
    transition_counts.items()
):

    print(
        f"  {transition}: {count}"
    )


print()
print("Observed durations:")

display_columns = [
    "project_id",
    "cycle_number",
    "transition",
    "start_event_date",
    "end_event_date",
    "duration_days",
]

print(
    observations_df[
        display_columns
    ].to_string(index=False)
)


# ============================================================
# Distribution summary
# ============================================================

print()
print("=" * 70)
print("DURATION DISTRIBUTION")
print("=" * 70)

distribution = (
    observations_df
    .groupby("transition")[
        "duration_days"
    ]
    .agg(
        count="count",
        min="min",
        median="median",
        mean="mean",
        max="max",
    )
    .reset_index()
)

print(
    distribution.to_string(
        index=False
    )
)


# ============================================================
# Important warning
# ============================================================

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "These are observed historical durations, "
    "not legal deadlines."
)

print(
    "They are NOT sufficient to establish "
    "national empirical benchmarks yet."
)

print(
    "The current dataset is suitable for "
    "pipeline validation and exploratory analysis, "
    "not production ML calibration."
)

print()
print(
    f"Output: {OUTPUT_FILE}"
)

print("=" * 70)