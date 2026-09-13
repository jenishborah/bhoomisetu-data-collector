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

FEATURES_FILE = (
    BASE_DIR
    / "project_features.csv"
)

TIMELINE_FILE = (
    BASE_DIR
    / "project_timeline.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "project_master.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_file(path):
    """
    Load a CSV file and fail clearly if it does not exist.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# NORMALIZE PROJECT ID
# ============================================================

def normalize_project_id(df):
    """
    Normalize project_id so joins are reliable.
    """

    df = df.copy()

    df["project_id"] = pd.to_numeric(
        df["project_id"],
        errors="coerce"
    )

    df = df[
        df["project_id"].notna()
    ]

    df["project_id"] = (
        df["project_id"]
        .astype(int)
    )

    return df


# ============================================================
# BUILD MASTER DATASET
# ============================================================

def build_master(
    features,
    timeline
):

    features = normalize_project_id(
        features
    )

    timeline = normalize_project_id(
        timeline
    )

    # --------------------------------------------------------
    # Check duplicate project IDs
    # --------------------------------------------------------

    feature_duplicates = (
        features["project_id"]
        .duplicated()
        .sum()
    )

    timeline_duplicates = (
        timeline["project_id"]
        .duplicated()
        .sum()
    )

    if feature_duplicates:

        raise ValueError(
            "project_features.csv contains "
            f"{feature_duplicates} duplicate project IDs."
        )

    if timeline_duplicates:

        raise ValueError(
            "project_timeline.csv contains "
            f"{timeline_duplicates} duplicate project IDs."
        )

    # --------------------------------------------------------
    # Avoid accidental duplicate columns
    # --------------------------------------------------------

    timeline_columns = [
        column
        for column in timeline.columns
        if column != "project_id"
    ]

    # --------------------------------------------------------
    # Left join
    # --------------------------------------------------------
    #
    # project_features is the master population.
    #
    # Every project should remain present even if timeline
    # information is incomplete.
    # --------------------------------------------------------

    master = features.merge(
        timeline[
            [
                "project_id"
            ]
            + timeline_columns
        ],
        on="project_id",
        how="left",
        validate="one_to_one"
    )

    return master


# ============================================================
# DATA QUALITY CHECKS
# ============================================================

def run_quality_checks(
    master
):

    print()
    print("=" * 70)
    print("DATA QUALITY CHECKS")
    print("=" * 70)

    # --------------------------------------------------------
    # Project count
    # --------------------------------------------------------

    print(
        f"Project rows       : {len(master)}"
    )

    print(
        f"Unique project IDs : "
        f"{master['project_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Duplicate IDs
    # --------------------------------------------------------

    duplicates = (
        master["project_id"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate IDs      : {duplicates}"
    )

    # --------------------------------------------------------
    # Missing timeline
    # --------------------------------------------------------

    missing_timeline = (
        master[
            master["first_event_date"]
            .isna()
        ]
    )

    print(
        f"Projects without timeline : "
        f"{len(missing_timeline)}"
    )

    # --------------------------------------------------------
    # Current/latest stage
    # --------------------------------------------------------

    if "latest_event_stage" in master:

        print()
        print(
            "Latest observed stage:"
        )

        print(
            master[
                "latest_event_stage"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    # --------------------------------------------------------
    # Acquisition completion
    # --------------------------------------------------------

    if "acquisition_completion_pct" in master:

        print()
        print(
            "Acquisition completion:"
        )

        print(
            master[
                [
                    "project_id",
                    "acquisition_completion_pct"
                ]
            ]
            .to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Missing-value overview
    # --------------------------------------------------------

    print()
    print(
        "Columns with missing values:"
    )

    missing = (
        master
        .isna()
        .sum()
    )

    missing = missing[
        missing > 0
    ].sort_values(
        ascending=False
    )

    if missing.empty:

        print("None")

    else:

        print(
            missing.to_string()
        )


# ============================================================
# DISPLAY MASTER DATASET
# ============================================================

def display_master(
    master
):

    print()
    print("=" * 70)
    print("PROJECT MASTER DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Compact project overview
    # --------------------------------------------------------

    overview_columns = [

        "project_id",

        "project_name",

        "state",

        "district",

        "land_required_ha",

        "land_to_acquire_ha",

        "land_acquired_ha",

        "acquisition_completion_pct",

        "survey_count",

        "party_record_count",

        "affected_party_count",

        "first_3a_date",

        "first_3A_date",

        "first_3D_date",

        "days_3a_to_3A",

        "days_3A_to_3D",

        "days_3a_to_3D",

        "latest_event_stage",
    ]

    available_columns = [
        column
        for column in overview_columns
        if column in master.columns
    ]

    print(
        master[
            available_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Dataset dimensions
    # --------------------------------------------------------

    print()
    print(
        f"Rows    : {master.shape[0]}"
    )

    print(
        f"Columns : {master.shape[1]}"
    )


# ============================================================
# SAVE
# ============================================================

def save_master(
    master
):

    master.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print(
        f"Saved: {OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BHOOMISETU PROJECT MASTER BUILDER")
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print(
        f"Loading: {FEATURES_FILE}"
    )

    features = load_file(
        FEATURES_FILE
    )

    print(
        f"Feature rows : {len(features)}"
    )

    print()

    print(
        f"Loading: {TIMELINE_FILE}"
    )

    timeline = load_file(
        TIMELINE_FILE
    )

    print(
        f"Timeline rows : {len(timeline)}"
    )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    master = build_master(
        features,
        timeline
    )

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    run_quality_checks(
        master
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    display_master(
        master
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_master(
        master
    )

    print()
    print("=" * 70)
    print(
        "PROJECT MASTER BUILD COMPLETED"
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