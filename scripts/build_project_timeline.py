from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = (
    Path("output")
    / "normalized"
    / "bhoomirashi"
)

OUTPUT_FILE = (
    BASE_DIR
    / "project_timeline.csv"
)


# ============================================================
# HELPERS
# ============================================================

STAGE_ORDER = {
    "3a": 1,
    "3A": 2,
    "3D": 3,
}


def load_notifications():
    """
    Load all project notification CSV files.
    """

    files = list(
        BASE_DIR.glob(
            "*/notifications.csv"
        )
    )

    dataframes = []

    for file in files:

        try:
            df = pd.read_csv(file)

        except Exception as exc:

            print(
                f"WARNING: Could not read "
                f"{file}: {exc}"
            )

            continue

        if df.empty:
            continue

        dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    df = pd.concat(
        dataframes,
        ignore_index=True
    )

    return df


# ============================================================
# NORMALIZE NOTIFICATIONS
# ============================================================

def normalize_notifications(df):
    """
    Normalize notification fields and dates.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Project ID
    # --------------------------------------------------------

    df["project_id"] = pd.to_numeric(
        df["project_id"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Notification ID
    # --------------------------------------------------------

    df["notification_id"] = pd.to_numeric(
        df["notification_id"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Publish date
    # --------------------------------------------------------

    df["publish_date"] = pd.to_datetime(
        df["publish_date"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Normalize notification type
    # --------------------------------------------------------

    df["notification_type"] = (
        df["notification_type"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Stage order
    # --------------------------------------------------------

    df["stage_order"] = (
        df["notification_type"]
        .map(STAGE_ORDER)
    )

    return df


# ============================================================
# BUILD PROJECT TIMELINE
# ============================================================

def build_project_timeline(df):
    """
    Build one project-level timeline row containing:

        - first/latest notification per stage
        - stage-to-stage durations
        - total notification span
        - current/latest observed stage
        - notification counts
    """

    records = []

    for project_id, group in df.groupby(
        "project_id"
    ):

        group = (
            group
            .sort_values(
                [
                    "publish_date",
                    "notification_id"
                ]
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Stage-specific subsets
        # ----------------------------------------------------

        stage_3a = group[
            group["notification_type"]
            .eq("3a")
        ]

        stage_3A = group[
            group["notification_type"]
            .eq("3A")
        ]

        stage_3D = group[
            group["notification_type"]
            .eq("3D")
        ]

        # ----------------------------------------------------
        # First / latest dates
        # ----------------------------------------------------

        first_3a = (
            stage_3a["publish_date"].min()
            if not stage_3a.empty
            else pd.NaT
        )

        latest_3a = (
            stage_3a["publish_date"].max()
            if not stage_3a.empty
            else pd.NaT
        )

        first_3A = (
            stage_3A["publish_date"].min()
            if not stage_3A.empty
            else pd.NaT
        )

        latest_3A = (
            stage_3A["publish_date"].max()
            if not stage_3A.empty
            else pd.NaT
        )

        first_3D = (
            stage_3D["publish_date"].min()
            if not stage_3D.empty
            else pd.NaT
        )

        latest_3D = (
            stage_3D["publish_date"].max()
            if not stage_3D.empty
            else pd.NaT
        )

        # ----------------------------------------------------
        # Helper for day differences
        # ----------------------------------------------------

        def days_between(
            start,
            end
        ):

            if (
                pd.isna(start)
                or
                pd.isna(end)
            ):
                return pd.NA

            return int(
                (end - start).days
            )

        # ----------------------------------------------------
        # Stage durations
        # ----------------------------------------------------

        days_3a_to_3A = days_between(
            first_3a,
            first_3A
        )

        days_3A_to_3D = days_between(
            first_3A,
            first_3D
        )

        days_3a_to_3D = days_between(
            first_3a,
            first_3D
        )

        # ----------------------------------------------------
        # Latest observed event
        # ----------------------------------------------------

        latest_event = group.iloc[-1]

        latest_event_date = (
            latest_event["publish_date"]
        )

        latest_event_stage = (
            latest_event["notification_type"]
        )

        latest_notification_id = (
            latest_event["notification_id"]
        )

        # ----------------------------------------------------
        # Notification counts
        # ----------------------------------------------------

        count_3a = len(stage_3a)
        count_3A = len(stage_3A)
        count_3D = len(stage_3D)

        total_notifications = len(group)

        # ----------------------------------------------------
        # Timeline span
        # ----------------------------------------------------

        first_event_date = (
            group["publish_date"].min()
        )

        last_event_date = (
            group["publish_date"].max()
        )

        timeline_span_days = days_between(
            first_event_date,
            last_event_date
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        records.append({

            "project_id":
                int(project_id),

            "first_event_date":
                first_event_date,

            "last_event_date":
                last_event_date,

            "latest_event_stage":
                latest_event_stage,

            "latest_notification_id":
                int(latest_notification_id),

            "notification_count_total":
                total_notifications,

            "notification_count_3a":
                count_3a,

            "notification_count_3A":
                count_3A,

            "notification_count_3D":
                count_3D,

            "first_3a_date":
                first_3a,

            "latest_3a_date":
                latest_3a,

            "first_3A_date":
                first_3A,

            "latest_3A_date":
                latest_3A,

            "first_3D_date":
                first_3D,

            "latest_3D_date":
                latest_3D,

            "days_3a_to_3A":
                days_3a_to_3A,

            "days_3A_to_3D":
                days_3A_to_3D,

            "days_3a_to_3D":
                days_3a_to_3D,

            "timeline_span_days":
                timeline_span_days,
        })

    return pd.DataFrame(records)


# ============================================================
# BUILD EVENT-LEVEL TIMELINE
# ============================================================

def build_event_timeline(df):
    """
    Create a clean event-level timeline.

    Unlike the project summary, this preserves every
    notification record.
    """

    event_df = df.copy()

    event_df = event_df.sort_values(
        [
            "project_id",
            "publish_date",
            "notification_id"
        ]
    )

    # --------------------------------------------------------
    # Days since previous notification within project
    # --------------------------------------------------------

    event_df[
        "previous_publish_date"
    ] = (
        event_df
        .groupby("project_id")[
            "publish_date"
        ]
        .shift(1)
    )

    event_df[
        "days_since_previous_event"
    ] = (
        event_df["publish_date"]
        -
        event_df["previous_publish_date"]
    ).dt.days

    # --------------------------------------------------------
    # Days since first event
    # --------------------------------------------------------

    first_dates = (
        event_df
        .groupby("project_id")[
            "publish_date"
        ]
        .transform("min")
    )

    event_df[
        "days_since_first_event"
    ] = (
        event_df["publish_date"]
        -
        first_dates
    ).dt.days

    # --------------------------------------------------------
    # Keep useful fields
    # --------------------------------------------------------

    columns = [

        "project_id",

        "notification_id",

        "notification_type",

        "stage_order",

        "serial_number",

        "publish_date",

        "notification_number",

        "status",

        "details_url",

        "objections_url",

        "previous_publish_date",

        "days_since_previous_event",

        "days_since_first_event",
    ]

    return event_df[
        columns
    ].reset_index(
        drop=True
    )


# ============================================================
# SAVE
# ============================================================

def save_csv(
    df,
    filename
):

    output_file = (
        BASE_DIR
        / filename
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "BHOOMISETU PROJECT TIMELINE BUILDER"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    notifications = (
        load_notifications()
    )

    print(
        f"Notification records loaded : "
        f"{len(notifications)}"
    )

    if notifications.empty:

        print()
        print(
            "ERROR: No notification data found."
        )

        return 1

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    notifications = (
        normalize_notifications(
            notifications
        )
    )

    # --------------------------------------------------------
    # Validate dates
    # --------------------------------------------------------

    invalid_dates = (
        notifications[
            notifications["publish_date"]
            .isna()
        ]
    )

    if not invalid_dates.empty:

        print()
        print(
            "WARNING: "
            f"{len(invalid_dates)} notification "
            "records have invalid dates."
        )

    # --------------------------------------------------------
    # Remove records with no project ID
    # --------------------------------------------------------

    notifications = (
        notifications[
            notifications["project_id"]
            .notna()
        ]
    )

    # --------------------------------------------------------
    # Build project summary
    # --------------------------------------------------------

    project_timeline = (
        build_project_timeline(
            notifications
        )
    )

    # --------------------------------------------------------
    # Build event timeline
    # --------------------------------------------------------

    event_timeline = (
        build_event_timeline(
            notifications
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    project_file = save_csv(
        project_timeline,
        "project_timeline.csv"
    )

    event_file = save_csv(
        event_timeline,
        "notification_timeline.csv"
    )

    # --------------------------------------------------------
    # Display project timeline
    # --------------------------------------------------------

    print()
    print(
        "PROJECT TIMELINE"
    )

    print("-" * 70)

    print(
        project_timeline.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Display event timeline
    # --------------------------------------------------------

    print()
    print(
        "NOTIFICATION EVENTS"
    )

    print("-" * 70)

    print(
        event_timeline[
            [
                "project_id",
                "notification_id",
                "notification_type",
                "publish_date",
                "days_since_previous_event",
                "days_since_first_event",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print(
        f"Projects generated : "
        f"{len(project_timeline)}"
    )

    print(
        f"Notification events : "
        f"{len(event_timeline)}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        project_file
    )

    print(
        event_file
    )

    print()

    print("=" * 70)
    print(
        "TIMELINE BUILD COMPLETED"
    )
    print("=" * 70)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        main()
    )