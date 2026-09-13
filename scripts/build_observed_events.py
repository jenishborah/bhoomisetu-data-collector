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

TIMELINE_FILE = DATA_DIR / "notification_timeline.csv"

OUTPUT_DIR = DATA_DIR / "temporal"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "observed_events.csv"


# ============================================================
# Load
# ============================================================

if not TIMELINE_FILE.exists():
    raise FileNotFoundError(
        f"Notification timeline not found: {TIMELINE_FILE}"
    )

df = pd.read_csv(TIMELINE_FILE)

print("=" * 70)
print("BhoomiSetu Observed Event Builder")
print("=" * 70)

print()
print(f"Input records: {len(df)}")


# ============================================================
# Validate required columns
# ============================================================

required_columns = [
    "project_id",
    "notification_id",
    "notification_type",
    "publish_date",
    "notification_number",
    "status",
    "details_url",
    "objections_url",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing)
    )


# ============================================================
# Normalize dates
# ============================================================

df["event_date"] = pd.to_datetime(
    df["publish_date"],
    errors="coerce",
)

invalid_dates = int(df["event_date"].isna().sum())

if invalid_dates:
    print(
        f"WARNING: {invalid_dates} records "
        "have invalid event dates."
    )


# ============================================================
# Normalize event types
# ============================================================

df["event_type"] = (
    df["notification_type"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# ============================================================
# Build canonical event table
# ============================================================

events = df[
    [
        "project_id",
        "event_date",
        "event_type",
        "notification_id",
        "notification_number",
        "status",
        "details_url",
        "objections_url",
    ]
].copy()


# ============================================================
# Sort
# ============================================================

events = events.sort_values(
    [
        "project_id",
        "event_date",
        "notification_id",
    ],
    na_position="last",
).reset_index(drop=True)


# ============================================================
# Event sequence
# ============================================================

events["event_sequence"] = (
    events.groupby("project_id")
    .cumcount()
    + 1
)


# ============================================================
# Days since previous event
# ============================================================

events["previous_event_date"] = (
    events.groupby("project_id")["event_date"]
    .shift(1)
)

events["days_since_previous_event"] = (
    events["event_date"]
    - events["previous_event_date"]
).dt.days


# ============================================================
# Days since first event
# ============================================================

events["first_event_date"] = (
    events.groupby("project_id")["event_date"]
    .transform("min")
)

events["days_since_first_event"] = (
    events["event_date"]
    - events["first_event_date"]
).dt.days


# ============================================================
# Stage transition indicators
# ============================================================

# IMPORTANT:
#
# BhoomiRashi uses:
#
#   3a  = Section 3a notification
#   3A  = Section 3A notification
#   3D  = Section 3D notification
#
# These are distinct event types.
#
# Do NOT convert event_type to lowercase/uppercase before
# comparing because "3a" and "3A" must remain separate.

events["is_3a"] = (
    events["event_type"] == "3a"
).astype(int)

events["is_3A"] = (
    events["event_type"] == "3A"
).astype(int)

events["is_3D"] = (
    events["event_type"] == "3D"
).astype(int)


# ============================================================
# Cumulative stage indicators
# ============================================================

# These represent whether the project has reached a stage
# by the current event date.
#
# They are useful for temporal snapshots because they use
# only information available up to that event.

events["has_3a_by_date"] = (
    events.groupby("project_id")["is_3a"]
    .cumsum()
    .clip(upper=1)
    .astype(int)
)

events["has_3A_by_date"] = (
    events.groupby("project_id")["is_3A"]
    .cumsum()
    .clip(upper=1)
    .astype(int)
)

events["has_3D_by_date"] = (
    events.groupby("project_id")["is_3D"]
    .cumsum()
    .clip(upper=1)
    .astype(int)
)


# ============================================================
# Stage reached by current event
# ============================================================

def determine_stage(row: pd.Series) -> str:
    """
    Determine the latest known statutory notification stage
    as of the current event.

    Priority:
        3D > 3A > 3a
    """

    if row["has_3D_by_date"] == 1:
        return "3D"

    if row["has_3A_by_date"] == 1:
        return "3A"

    if row["has_3a_by_date"] == 1:
        return "3a"

    return "UNKNOWN"


events["stage_as_of_event"] = events.apply(
    determine_stage,
    axis=1,
)


# ============================================================
# Validation
# ============================================================

duplicate_event_keys = int(
    events.duplicated(
        subset=[
            "project_id",
            "notification_id",
        ]
    ).sum()
)

if duplicate_event_keys:
    print(
        "WARNING: duplicate project_id + "
        f"notification_id records: "
        f"{duplicate_event_keys}"
    )


# ============================================================
# Stage count validation
# ============================================================

count_3a = int(events["is_3a"].sum())
count_3A = int(events["is_3A"].sum())
count_3D = int(events["is_3D"].sum())

print()
print("Stage event counts:")
print(f"  3a : {count_3a}")
print(f"  3A : {count_3A}")
print(f"  3D : {count_3D}")

print()
print("Expected total:")
print(
    f"  {count_3a} + {count_3A} + {count_3D} "
    f"= {count_3a + count_3A + count_3D}"
)

if (
    count_3a
    + count_3A
    + count_3D
    != len(events)
):

    print(
        "WARNING: Some records have an event type "
        "other than 3a, 3A, or 3D."
    )


# ============================================================
# Save
# ============================================================

events.to_csv(
    OUTPUT_FILE,
    index=False,
)

print()
print("Output:")
print(f"  {OUTPUT_FILE}")

print()
print("Projects:")
print(
    f"  {events['project_id'].nunique()}"
)

print("Events:")
print(
    f"  {len(events)}"
)

print()
print("Events by project:")

project_counts = (
    events.groupby("project_id")
    .size()
    .sort_index()
)

for project_id, count in project_counts.items():

    print(
        f"  {project_id}: {count}"
    )

print()
print("Latest stage by project:")

latest_stage = (
    events
    .sort_values(
        ["project_id", "event_date", "notification_id"]
    )
    .groupby("project_id")
    .tail(1)
    [["project_id", "stage_as_of_event"]]
    .sort_values("project_id")
)

for _, row in latest_stage.iterrows():

    print(
        f"  {row['project_id']}: "
        f"{row['stage_as_of_event']}"
    )

print()
print("=" * 70)
print("Observed event dataset created successfully.")
print("=" * 70)