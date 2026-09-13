import csv
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
    / "project_features.csv"
)


# ============================================================
# HELPERS
# ============================================================

def load_csv_files(pattern):
    """
    Load all matching CSV files recursively.

    Empty files / files containing only headers are ignored.
    """

    files = list(BASE_DIR.glob(pattern))

    dataframes = []

    for file in files:

        try:
            df = pd.read_csv(file)

        except Exception as exc:
            print(
                f"WARNING: Could not read {file}: {exc}"
            )
            continue

        if df.empty:
            continue

        dataframes.append(df)

    if not dataframes:
        return pd.DataFrame()

    return pd.concat(
        dataframes,
        ignore_index=True
    )


def safe_divide(numerator, denominator):
    """
    Safe division returning 0 when denominator is zero.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator


# ============================================================
# LOAD PROJECT MANIFESTS
# ============================================================

def load_project_manifests():
    """
    Load project_manifest.csv files from all projects.
    """

    files = list(
        BASE_DIR.glob(
            "*/project_manifest.csv"
        )
    )

    records = []

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

        records.append(df)

    if not records:
        return pd.DataFrame()

    return pd.concat(
        records,
        ignore_index=True
    )


# ============================================================
# LOAD 3D SURVEY DATA
# ============================================================

def load_surveys():
    """
    Load all normalized 3D survey files.
    """

    return load_csv_files(
        "*/3d_*_surveys.csv"
    )


# ============================================================
# LOAD 3D PARTY DATA
# ============================================================

def load_parties():
    """
    Load all normalized 3D land-party files.
    """

    return load_csv_files(
        "*/3d_*_land_parties.csv"
    )


# ============================================================
# BUILD SURVEY FEATURES
# ============================================================

def build_survey_features(surveys):
    """
    Aggregate survey-level data into project-level features.
    """

    if surveys.empty:
        return pd.DataFrame()

    records = []

    for project_id, group in surveys.groupby(
        "project_id"
    ):

        surveyed_area = (
            group["area_hectares"]
            .sum()
        )

        private_area = (
            group.loc[
                group["land_nature"]
                .eq("Private"),
                "area_hectares"
            ]
            .sum()
        )

        government_area = (
            group.loc[
                group["land_nature"]
                .eq("Government"),
                "area_hectares"
            ]
            .sum()
        )

        survey_count = len(group)

        urban_count = (
            group["land_category"]
            .eq("Urban")
            .sum()
        )

        wet_count = (
            group["land_type"]
            .eq("Wet")
            .sum()
        )

        records.append({

            "project_id":
                int(project_id),

            "survey_count":
                int(survey_count),

            "surveyed_area_ha":
                float(surveyed_area),

            "private_area_ha":
                float(private_area),

            "government_area_ha":
                float(government_area),

            "private_land_pct":
                safe_divide(
                    private_area,
                    surveyed_area
                ) * 100,

            "government_land_pct":
                safe_divide(
                    government_area,
                    surveyed_area
                ) * 100,

            "village_count":
                int(
                    group["village"]
                    .nunique()
                ),

            "district_count":
                int(
                    group["district"]
                    .nunique()
                ),

            "urban_survey_count":
                int(urban_count),

            "urban_survey_pct":
                safe_divide(
                    urban_count,
                    survey_count
                ) * 100,

            "wet_survey_count":
                int(wet_count),

            "wet_land_pct":
                safe_divide(
                    wet_count,
                    survey_count
                ) * 100,
        })

    return pd.DataFrame(records)


# ============================================================
# BUILD PARTY FEATURES
# ============================================================

def build_party_features(parties):
    """
    Aggregate party-level ownership and stakeholder
    complexity into project-level features.
    """

    if parties.empty:
        return pd.DataFrame()

    records = []

    for project_id, group in parties.groupby(
        "project_id"
    ):

        party_count = len(group)

        owner_count = (
            group["party_type"]
            .eq("Owner")
            .sum()
        )

        affected_count = (
            group["party_type"]
            .eq("Affected Party")
            .sum()
        )

        numeric_area = (
            pd.to_numeric(
                group["party_area_hectares"],
                errors="coerce"
            )
        )

        records.append({

            "project_id":
                int(project_id),

            "party_record_count":
                int(party_count),

            "owner_record_count":
                int(owner_count),

            "affected_party_count":
                int(affected_count),

            "affected_party_rate_pct":
                safe_divide(
                    affected_count,
                    party_count
                ) * 100,

            "party_area_numeric_count":
                int(
                    numeric_area
                    .notna()
                    .sum()
                ),

            "party_area_missing_count":
                int(
                    numeric_area
                    .isna()
                    .sum()
                ),

            "party_area_total_ha":
                float(
                    numeric_area
                    .sum(
                        min_count=1
                    )
                ),

            "party_area_mean_ha":
                float(
                    numeric_area
                    .mean()
                ),

            "party_area_max_ha":
                float(
                    numeric_area
                    .max()
                ),
        })

    return pd.DataFrame(records)


# ============================================================
# BUILD PROJECT FEATURES
# ============================================================

def build_project_features():

    print("=" * 70)
    print(
        "BHOOMISETU PROJECT FEATURE BUILDER"
    )
    print("=" * 70)

    print()

    # --------------------------------------------------------
    # Load manifests
    # --------------------------------------------------------

    manifests = (
        load_project_manifests()
    )

    print(
        f"Project manifest rows : "
        f"{len(manifests)}"
    )

    # --------------------------------------------------------
    # Load surveys
    # --------------------------------------------------------

    surveys = load_surveys()

    print(
        f"3D survey rows       : "
        f"{len(surveys)}"
    )

    # --------------------------------------------------------
    # Load parties
    # --------------------------------------------------------

    parties = load_parties()

    print(
        f"3D party rows        : "
        f"{len(parties)}"
    )

    print()

    # --------------------------------------------------------
    # Build component features
    # --------------------------------------------------------

    survey_features = (
        build_survey_features(
            surveys
        )
    )

    party_features = (
        build_party_features(
            parties
        )
    )

    # --------------------------------------------------------
    # Start with manifest data
    # --------------------------------------------------------

    if manifests.empty:

        if survey_features.empty:

            print(
                "ERROR: No project data found."
            )

            return 1

        project_features = (
            survey_features.copy()
        )

    else:

        project_features = (
            manifests.copy()
        )

        # ----------------------------------------------------
        # Normalize project ID
        # ----------------------------------------------------

        project_features[
            "project_id"
        ] = pd.to_numeric(
            project_features[
                "project_id"
            ],
            errors="coerce"
        )

        # ----------------------------------------------------
        # Remove duplicate project rows
        # ----------------------------------------------------

        project_features = (
            project_features
            .drop_duplicates(
                subset=["project_id"]
            )
        )

        # ----------------------------------------------------
        # Join survey features
        # ----------------------------------------------------

        if not survey_features.empty:

            project_features = (
                project_features.merge(
                    survey_features,
                    on="project_id",
                    how="left"
                )
            )

        # ----------------------------------------------------
        # Join party features
        # ----------------------------------------------------

        if not party_features.empty:

            project_features = (
                project_features.merge(
                    party_features,
                    on="project_id",
                    how="left"
                )
            )

    # --------------------------------------------------------
    # Fill count features where there is genuinely no
    # corresponding 3D detail.
    #
    # IMPORTANT:
    # These zeros mean "no extracted 3D detail records",
    # NOT "the project definitely has zero such records".
    # --------------------------------------------------------

    count_columns = [
        "survey_count",
        "village_count",
        "district_count",
        "urban_survey_count",
        "wet_survey_count",
        "party_record_count",
        "owner_record_count",
        "affected_party_count",
        "party_area_numeric_count",
        "party_area_missing_count",
    ]

    for column in count_columns:

        if column in project_features.columns:

            project_features[column] = (
                project_features[column]
                .fillna(0)
                .astype(int)
            )

    # --------------------------------------------------------
    # Preserve missing area aggregates where no 3D party
    # records exist.
    # Do NOT convert these to zero.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Derived acquisition completion
    # --------------------------------------------------------

    if (
        "land_to_acquire_ha"
        in project_features.columns
        and
        "land_acquired_ha"
        in project_features.columns
    ):

        project_features[
            "acquisition_completion_pct"
        ] = (
            project_features[
                "land_acquired_ha"
            ]
            /
            project_features[
                "land_to_acquire_ha"
            ]
            .replace(0, pd.NA)
        ) * 100

    # --------------------------------------------------------
    # Derived survey coverage
    # --------------------------------------------------------

    if (
        "land_to_acquire_ha"
        in project_features.columns
        and
        "surveyed_area_ha"
        in project_features.columns
    ):

        project_features[
            "surveyed_vs_land_to_acquire_pct"
        ] = (
            project_features[
                "surveyed_area_ha"
            ]
            /
            project_features[
                "land_to_acquire_ha"
            ]
            .replace(0, pd.NA)
        ) * 100

    # --------------------------------------------------------
    # Complexity ratios
    # --------------------------------------------------------

    if (
        "party_record_count"
        in project_features.columns
        and
        "survey_count"
        in project_features.columns
    ):

        project_features[
            "parties_per_survey"
        ] = (
            project_features[
                "party_record_count"
            ]
            /
            project_features[
                "survey_count"
            ]
            .replace(0, pd.NA)
        )

    if (
        "owner_record_count"
        in project_features.columns
        and
        "survey_count"
        in project_features.columns
    ):

        project_features[
            "owners_per_survey"
        ] = (
            project_features[
                "owner_record_count"
            ]
            /
            project_features[
                "survey_count"
            ]
            .replace(0, pd.NA)
        )

    # --------------------------------------------------------
    # Reorder project_id first
    # --------------------------------------------------------

    if "project_id" in project_features.columns:

        columns = (
            ["project_id"]
            +
            [
                column
                for column in project_features.columns
                if column != "project_id"
            ]
        )

        project_features = (
            project_features[
                columns
            ]
        )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    project_features = (
        project_features
        .sort_values(
            "project_id"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    BASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    project_features.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print()
    print(
        "PROJECT FEATURES"
    )

    print("-" * 70)

    print(
        project_features.to_string(
            index=False
        )
    )

    print()

    print(
        f"Projects generated : "
        f"{len(project_features)}"
    )

    print(
        f"Features generated : "
        f"{len(project_features.columns)}"
    )

    print()

    print(
        "Saved:"
    )

    print(
        OUTPUT_FILE
    )

    print()

    print("=" * 70)
    print(
        "FEATURE BUILD COMPLETED"
    )
    print("=" * 70)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    raise SystemExit(
        build_project_features()
    )