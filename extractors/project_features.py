from pathlib import Path
import csv
from datetime import datetime
from statistics import mean, median


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "54635"

DATA_DIR = (
    Path("output")
    / "normalized"
    / "bhoomirashi"
    / PROJECT_ID
)

PROJECT_FILE = DATA_DIR / "project_manifest.csv"
NOTIFICATIONS_FILE = DATA_DIR / "notifications.csv"
TIMELINE_FILE = DATA_DIR / "stage_timeline.csv"
SURVEYS_FILE = DATA_DIR / "acquisition_surveys.csv"
PARTIES_FILES = [
    DATA_DIR / "3d_34193_land_parties.csv",
    DATA_DIR / "3d_35100_land_parties.csv",
    DATA_DIR / "3d_44312_land_parties.csv",
]

OUTPUT_FILE = DATA_DIR / "project_features.csv"


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))


def to_float(value):
    if value is None or value == "":
        return None

    return float(value)


def to_int(value):
    if value is None or value == "":
        return None

    return int(float(value))


def parse_date(value):
    if not value:
        return None

    return datetime.strptime(
        value,
        "%Y-%m-%d"
    ).date()


def safe_percentage(numerator, denominator):
    if denominator is None or denominator == 0:
        return None

    return (numerator / denominator) * 100.0


def round_value(value, digits=6):
    if value is None:
        return None

    return round(value, digits)


# ============================================================
# PROJECT FEATURES
# ============================================================

def build_project_features():

    # --------------------------------------------------------
    # Load source files
    # --------------------------------------------------------

    project_rows = load_csv(PROJECT_FILE)
    notification_rows = load_csv(NOTIFICATIONS_FILE)
    timeline_rows = load_csv(TIMELINE_FILE)
    survey_rows = load_csv(SURVEYS_FILE)

    party_rows = []

    for party_file in PARTIES_FILES:
        party_rows.extend(
            load_csv(party_file)
        )

    if len(project_rows) != 1:
        raise ValueError(
            f"Expected exactly one project row, "
            f"found {len(project_rows)}"
        )

    project = project_rows[0]

    # --------------------------------------------------------
    # Validate project ID
    # --------------------------------------------------------

    if project["project_id"] != PROJECT_ID:
        raise ValueError(
            f"Unexpected project ID: "
            f"{project['project_id']}"
        )

    # ========================================================
    # 1. LAND / ACQUISITION PROGRESS
    # ========================================================

    land_required_ha = to_float(
        project["land_required_ha"]
    )

    land_available_ha = to_float(
        project["land_available_ha"]
    )

    land_to_acquire_ha = to_float(
        project["land_to_acquire_ha"]
    )

    land_acquired_ha = to_float(
        project["land_acquired_ha"]
    )

    land_remaining_ha = None

    if (
        land_to_acquire_ha is not None
        and land_acquired_ha is not None
    ):
        land_remaining_ha = max(
            land_to_acquire_ha - land_acquired_ha,
            0.0
        )

    acquisition_completion_pct = safe_percentage(
        land_acquired_ha,
        land_to_acquire_ha
    )

    # ========================================================
    # 2. SURVEY / 3D COVERAGE
    # ========================================================

    total_surveys = len(survey_rows)

    surveyed_area_ha = sum(
        to_float(row["area_hectares"])
        for row in survey_rows
        if row["area_hectares"] != ""
    )

    surveyed_vs_required_pct = safe_percentage(
        surveyed_area_ha,
        land_required_ha
    )

    surveyed_vs_acquisition_target_pct = safe_percentage(
        surveyed_area_ha,
        land_to_acquire_ha
    )

    # --------------------------------------------------------
    # Private / Government land
    # --------------------------------------------------------

    private_area_ha = sum(
        to_float(row["area_hectares"])
        for row in survey_rows
        if row.get("land_nature") == "Private"
        and row["area_hectares"] != ""
    )

    government_area_ha = sum(
        to_float(row["area_hectares"])
        for row in survey_rows
        if row.get("land_nature") == "Government"
        and row["area_hectares"] != ""
    )

    private_land_pct = safe_percentage(
        private_area_ha,
        surveyed_area_ha
    )

    government_land_pct = safe_percentage(
        government_area_ha,
        surveyed_area_ha
    )

    # ========================================================
    # 3. GEOGRAPHIC / VILLAGE COMPLEXITY
    # ========================================================

    villages = {
        row.get("village", "").strip()
        for row in survey_rows
        if row.get("village", "").strip()
    }

    districts = {
        row.get("district", "").strip()
        for row in survey_rows
        if row.get("district", "").strip()
    }

    sub_districts = {
        row.get("sub_district", "").strip()
        for row in survey_rows
        if row.get("sub_district", "").strip()
    }

    land_types = {
        row.get("land_type", "").strip()
        for row in survey_rows
        if row.get("land_type", "").strip()
    }

    land_categories = {
        row.get("land_category", "").strip()
        for row in survey_rows
        if row.get("land_category", "").strip()
    }

    # ========================================================
    # 4. STAKEHOLDER COMPLEXITY
    # ========================================================

    owner_count = sum(
        1
        for row in party_rows
        if row.get("party_type") == "Owner"
    )

    affected_party_count = sum(
        1
        for row in party_rows
        if row.get("party_type") == "Affected Party"
    )

    party_count = len(party_rows)

    affected_party_density = safe_percentage(
        affected_party_count,
        total_surveys
    )

    owners_per_survey = None

    if total_surveys > 0:
        owners_per_survey = (
            owner_count / total_surveys
        )

    # ========================================================
    # 5. NOTIFICATION COMPLEXITY
    # ========================================================

    notification_3a_count = sum(
        1
        for row in notification_rows
        if row.get("notification_type") == "3a"
    )

    notification_3A_count = sum(
        1
        for row in notification_rows
        if row.get("notification_type") == "3A"
    )

    notification_3D_count = sum(
        1
        for row in notification_rows
        if row.get("notification_type") == "3D"
    )

    total_notification_count = len(
        notification_rows
    )

    # Number of distinct notification cycles.
    #
    # A cycle is treated as a sequence beginning with 3a.
    # We are NOT claiming this is a legal acquisition cycle;
    # it is only an analytical grouping.
    notification_cycle_count = (
        notification_3a_count
    )

    # ========================================================
    # 6. TEMPORAL FEATURES
    # ========================================================

    timeline_dates = []

    for row in timeline_rows:

        event_date = parse_date(
            row["event_date"]
        )

        if event_date is not None:
            timeline_dates.append(event_date)

    timeline_dates.sort()

    first_event_date = (
        timeline_dates[0]
        if timeline_dates
        else None
    )

    latest_event_date = (
        timeline_dates[-1]
        if timeline_dates
        else None
    )

    timeline_span_days = None

    if (
        first_event_date is not None
        and latest_event_date is not None
    ):
        timeline_span_days = (
            latest_event_date
            - first_event_date
        ).days

    # --------------------------------------------------------
    # Event gaps
    # --------------------------------------------------------

    event_gaps = []

    previous_date = None

    for date in timeline_dates:

        if previous_date is not None:

            event_gaps.append(
                (date - previous_date).days
            )

        previous_date = date

    average_event_gap_days = (
        mean(event_gaps)
        if event_gaps
        else None
    )

    median_event_gap_days = (
        median(event_gaps)
        if event_gaps
        else None
    )

    max_event_gap_days = (
        max(event_gaps)
        if event_gaps
        else None
    )

    # --------------------------------------------------------
    # Notification-specific dates
    # --------------------------------------------------------

    notifications = []

    for row in notification_rows:

        notifications.append({
            "type": row["notification_type"],
            "date": parse_date(
                row["publish_date"]
            ),
        })

    notifications.sort(
        key=lambda x: x["date"]
    )

    preliminary_3a_dates = [
        n["date"]
        for n in notifications
        if n["type"] == "3a"
        and n["date"] is not None
    ]

    declaration_3A_dates = [
        n["date"]
        for n in notifications
        if n["type"] == "3A"
        and n["date"] is not None
    ]

    declaration_3D_dates = [
        n["date"]
        for n in notifications
        if n["type"] == "3D"
        and n["date"] is not None
    ]

    first_3a_date = (
        min(preliminary_3a_dates)
        if preliminary_3a_dates
        else None
    )

    latest_3a_date = (
        max(preliminary_3a_dates)
        if preliminary_3a_dates
        else None
    )

    first_3A_date = (
        min(declaration_3A_dates)
        if declaration_3A_dates
        else None
    )

    latest_3A_date = (
        max(declaration_3A_dates)
        if declaration_3A_dates
        else None
    )

    first_3D_date = (
        min(declaration_3D_dates)
        if declaration_3D_dates
        else None
    )

    latest_3D_date = (
        max(declaration_3D_dates)
        if declaration_3D_dates
        else None
    )

    # --------------------------------------------------------
    # Stage transition gaps
    # --------------------------------------------------------

    days_first_3a_to_first_3A = None

    if (
        first_3a_date is not None
        and first_3A_date is not None
    ):
        days_first_3a_to_first_3A = (
            first_3A_date
            - first_3a_date
        ).days

    days_first_3A_to_first_3D = None

    if (
        first_3A_date is not None
        and first_3D_date is not None
    ):
        days_first_3A_to_first_3D = (
            first_3D_date
            - first_3A_date
        ).days

    days_first_3a_to_first_3D = None

    if (
        first_3a_date is not None
        and first_3D_date is not None
    ):
        days_first_3a_to_first_3D = (
            first_3D_date
            - first_3a_date
        ).days

    days_latest_3a_to_latest_3D = None

    if (
        latest_3a_date is not None
        and latest_3D_date is not None
    ):
        days_latest_3a_to_latest_3D = (
            latest_3D_date
            - latest_3a_date
        ).days

    # --------------------------------------------------------
    # Latest 3D → latest known event
    # --------------------------------------------------------

    days_since_latest_3D_event = None

    if (
        latest_3D_date is not None
        and latest_event_date is not None
    ):
        days_since_latest_3D_event = (
            latest_event_date
            - latest_3D_date
        ).days

    # ========================================================
    # 7. SURVEY-LEVEL AGGREGATES
    # ========================================================

    total_owner_count_from_surveys = sum(
        to_int(row["owner_count"])
        for row in survey_rows
        if row.get("owner_count", "") != ""
    )

    total_affected_from_surveys = sum(
        to_int(row["affected_party_count"])
        for row in survey_rows
        if row.get("affected_party_count", "") != ""
    )

    # ========================================================
    # 8. DATA QUALITY SIGNALS
    # ========================================================

    missing_party_area_count = sum(
        1
        for row in party_rows
        if not row.get("party_area_hectares")
    )

    numeric_party_area_count = (
        party_count
        - missing_party_area_count
    )

    # ========================================================
    # BUILD FINAL FEATURE RECORD
    # ========================================================

    features = {

        # ----------------------------------------------------
        # Identity
        # ----------------------------------------------------

        "project_id": PROJECT_ID,

        # ----------------------------------------------------
        # Raw project facts
        # ----------------------------------------------------

        "project_number": project.get(
            "project_number",
            ""
        ),

        "land_required_ha": land_required_ha,

        "land_available_ha": land_available_ha,

        "land_to_acquire_ha": land_to_acquire_ha,

        "land_acquired_ha": land_acquired_ha,

        # ----------------------------------------------------
        # Derived acquisition progress
        # ----------------------------------------------------

        "land_remaining_ha": round_value(
            land_remaining_ha
        ),

        "acquisition_completion_pct": round_value(
            acquisition_completion_pct,
            4
        ),

        # ----------------------------------------------------
        # Survey / 3D coverage
        # ----------------------------------------------------

        "total_surveys": total_surveys,

        "surveyed_area_ha": round_value(
            surveyed_area_ha
        ),

        "surveyed_vs_required_pct": round_value(
            surveyed_vs_required_pct,
            4
        ),

        "surveyed_vs_acquisition_target_pct": round_value(
            surveyed_vs_acquisition_target_pct,
            4
        ),

        "private_area_ha": round_value(
            private_area_ha
        ),

        "government_area_ha": round_value(
            government_area_ha
        ),

        "private_land_pct": round_value(
            private_land_pct,
            4
        ),

        "government_land_pct": round_value(
            government_land_pct,
            4
        ),

        # ----------------------------------------------------
        # Geography
        # ----------------------------------------------------

        "village_count": len(villages),

        "district_count": len(districts),

        "sub_district_count": len(sub_districts),

        "land_type_count": len(land_types),

        "land_category_count": len(
            land_categories
        ),

        # ----------------------------------------------------
        # Stakeholder complexity
        # ----------------------------------------------------

        "total_party_records": party_count,

        "total_owners": owner_count,

        "total_affected_parties": affected_party_count,

        "affected_party_density_pct": round_value(
            affected_party_density,
            4
        ),

        "owners_per_survey": round_value(
            owners_per_survey,
            4
        ),

        "survey_owner_count_sum": (
            total_owner_count_from_surveys
        ),

        "survey_affected_party_count_sum": (
            total_affected_from_surveys
        ),

        # ----------------------------------------------------
        # Notification complexity
        # ----------------------------------------------------

        "notification_3a_count": (
            notification_3a_count
        ),

        "notification_3A_count": (
            notification_3A_count
        ),

        "notification_3D_count": (
            notification_3D_count
        ),

        "total_notification_count": (
            total_notification_count
        ),

        "notification_cycle_count": (
            notification_cycle_count
        ),

        # ----------------------------------------------------
        # Temporal features
        # ----------------------------------------------------

        "first_event_date": (
            first_event_date.isoformat()
            if first_event_date
            else ""
        ),

        "latest_event_date": (
            latest_event_date.isoformat()
            if latest_event_date
            else ""
        ),

        "timeline_span_days": (
            timeline_span_days
        ),

        "average_event_gap_days": round_value(
            average_event_gap_days,
            4
        ),

        "median_event_gap_days": round_value(
            median_event_gap_days,
            4
        ),

        "max_event_gap_days": (
            max_event_gap_days
        ),

        # ----------------------------------------------------
        # Stage transition features
        # ----------------------------------------------------

        "first_3a_date": (
            first_3a_date.isoformat()
            if first_3a_date
            else ""
        ),

        "first_3A_date": (
            first_3A_date.isoformat()
            if first_3A_date
            else ""
        ),

        "first_3D_date": (
            first_3D_date.isoformat()
            if first_3D_date
            else ""
        ),

        "latest_3a_date": (
            latest_3a_date.isoformat()
            if latest_3a_date
            else ""
        ),

        "latest_3A_date": (
            latest_3A_date.isoformat()
            if latest_3A_date
            else ""
        ),

        "latest_3D_date": (
            latest_3D_date.isoformat()
            if latest_3D_date
            else ""
        ),

        "days_first_3a_to_first_3A": (
            days_first_3a_to_first_3A
        ),

        "days_first_3A_to_first_3D": (
            days_first_3A_to_first_3D
        ),

        "days_first_3a_to_first_3D": (
            days_first_3a_to_first_3D
        ),

        "days_latest_3a_to_latest_3D": (
            days_latest_3a_to_latest_3D
        ),

        "days_since_latest_3D_event": (
            days_since_latest_3D_event
        ),

        # ----------------------------------------------------
        # Data quality
        # ----------------------------------------------------

        "party_area_numeric_count": (
            numeric_party_area_count
        ),

        "party_area_missing_count": (
            missing_party_area_count
        ),
    }

    return features


# ============================================================
# SAVE
# ============================================================

def save_features(features):

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = list(
        features.keys()
    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()
        writer.writerow(features)

    return OUTPUT_FILE


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(features):

    print()
    print("=" * 70)
    print("BHOOMISETU PROJECT FEATURE SUMMARY")
    print("=" * 70)

    sections = {

        "ACQUISITION PROGRESS": [
            "land_required_ha",
            "land_to_acquire_ha",
            "land_acquired_ha",
            "land_remaining_ha",
            "acquisition_completion_pct",
        ],

        "3D SURVEY COVERAGE": [
            "total_surveys",
            "surveyed_area_ha",
            "surveyed_vs_required_pct",
            "surveyed_vs_acquisition_target_pct",
            "private_area_ha",
            "government_area_ha",
            "private_land_pct",
            "government_land_pct",
        ],

        "STAKEHOLDER COMPLEXITY": [
            "total_party_records",
            "total_owners",
            "total_affected_parties",
            "affected_party_density_pct",
            "owners_per_survey",
        ],

        "NOTIFICATION COMPLEXITY": [
            "notification_3a_count",
            "notification_3A_count",
            "notification_3D_count",
            "total_notification_count",
            "notification_cycle_count",
        ],

        "TEMPORAL SIGNALS": [
            "timeline_span_days",
            "average_event_gap_days",
            "median_event_gap_days",
            "max_event_gap_days",
            "days_first_3a_to_first_3A",
            "days_first_3A_to_first_3D",
            "days_first_3a_to_first_3D",
            "days_latest_3a_to_latest_3D",
            "days_since_latest_3D_event",
        ],

        "DATA QUALITY": [
            "party_area_numeric_count",
            "party_area_missing_count",
        ],
    }

    for section, fields in sections.items():

        print()
        print(section)
        print("-" * 70)

        for field in fields:

            print(
                f"{field:40}: "
                f"{features.get(field)}"
            )

    print()
    print("=" * 70)
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("BhoomiSetu Project Feature Engineering")
    print("=" * 70)

    print(
        f"Project ID: {PROJECT_ID}"
    )

    print(
        f"Data directory: {DATA_DIR}"
    )

    features = build_project_features()

    save_features(features)

    print_summary(features)


if __name__ == "__main__":
    main()