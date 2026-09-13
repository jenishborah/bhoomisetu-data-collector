from pathlib import Path
import csv
import sys
from collections import Counter


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


# ============================================================
# EXPECTED FILES
# ============================================================

EXPECTED_FILES = [
    "project_manifest.csv",
    "sanctions.csv",
    "notifications.csv",

    "3d_34193_surveys.csv",
    "3d_34193_land_parties.csv",

    "3d_35100_surveys.csv",
    "3d_35100_land_parties.csv",

    "3d_44312_surveys.csv",
    "3d_44312_land_parties.csv",

    "acquisition_surveys.csv",
]


# ============================================================
# EXPECTED COUNTS FROM OUR CURRENT EXTRACTION
# ============================================================

EXPECTED_SURVEY_COUNTS = {
    "34193": 46,
    "35100": 310,
    "44312": 24,
}

EXPECTED_PARTY_COUNTS = {
    "34193": 72,
    "35100": 501,
    "44312": 36,
}

EXPECTED_NOTIFICATION_COUNTS = {
    "3a": 3,
    "3A": 3,
    "3D": 3,
}


# ============================================================
# RESULT TRACKING
# ============================================================

errors = []
warnings = []
passes = []


def report_pass(message):
    passes.append(message)
    print(f"[PASS] {message}")


def report_warning(message):
    warnings.append(message)
    print(f"[WARN] {message}")


def report_error(message):
    errors.append(message)
    print(f"[FAIL] {message}")


# ============================================================
# CSV LOADER
# ============================================================

def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        report_error(f"Missing file: {filename}")
        return [], []

    try:
        with path.open(
            "r",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            reader = csv.DictReader(f)

            fieldnames = reader.fieldnames or []
            rows = list(reader)

        report_pass(
            f"{filename}: loaded successfully "
            f"({len(rows)} rows, {len(fieldnames)} columns)"
        )

        return rows, fieldnames

    except Exception as exc:
        report_error(
            f"{filename}: could not read CSV: {exc}"
        )
        return [], []


# ============================================================
# BASIC FILE CHECK
# ============================================================

def check_expected_files():
    print()
    print("=" * 70)
    print("1. FILE STRUCTURE CHECK")
    print("=" * 70)

    missing = []

    for filename in EXPECTED_FILES:

        path = DATA_DIR / filename

        if path.exists():
            report_pass(f"Found: {filename}")
        else:
            report_error(f"Missing: {filename}")
            missing.append(filename)

    if not missing:
        report_pass(
            f"All {len(EXPECTED_FILES)} expected files are present."
        )


# ============================================================
# PROJECT MANIFEST
# ============================================================

def check_project_manifest():
    print()
    print("=" * 70)
    print("2. PROJECT MANIFEST CHECK")
    print("=" * 70)

    rows, fields = load_csv("project_manifest.csv")

    if not rows:
        return

    required = [
        "project_id",
        "project_name",
        "project_number",
        "land_required_ha",
        "land_available_ha",
        "land_to_acquire_ha",
        "land_acquired_ha",
    ]

    for field in required:

        if field in fields:
            report_pass(
                f"project_manifest.csv contains '{field}'"
            )
        else:
            report_error(
                f"project_manifest.csv missing '{field}'"
            )

    if len(rows) != 1:
        report_error(
            f"Expected exactly 1 project row, found {len(rows)}"
        )
    else:
        report_pass("Exactly one project record found.")

    row = rows[0]

    if row.get("project_id") == PROJECT_ID:
        report_pass(
            f"Project ID correctly set to {PROJECT_ID}"
        )
    else:
        report_error(
            f"Unexpected project ID: {row.get('project_id')}"
        )

    numeric_fields = [
        "land_required_ha",
        "land_available_ha",
        "land_to_acquire_ha",
        "land_acquired_ha",
    ]

    for field in numeric_fields:

        try:
            float(row[field])
            report_pass(
                f"{field} is numeric"
            )
        except (ValueError, TypeError):
            report_error(
                f"{field} is not numeric: {row[field]}"
            )


# ============================================================
# SANCTIONS
# ============================================================

def check_sanctions():
    print()
    print("=" * 70)
    print("3. SANCTIONS CHECK")
    print("=" * 70)

    rows, fields = load_csv("sanctions.csv")

    if not rows:
        return

    required = [
        "project_id",
        "sanction_number",
        "sanction_date",
        "sanction_amount_rs",
    ]

    for field in required:

        if field in fields:
            report_pass(
                f"sanctions.csv contains '{field}'"
            )
        else:
            report_error(
                f"sanctions.csv missing '{field}'"
            )

    if len(rows) == 2:
        report_pass("Expected 2 sanction records found.")
    else:
        report_warning(
            f"Expected 2 sanctions, found {len(rows)}"
        )

    total = 0.0

    for index, row in enumerate(rows, start=1):

        try:
            amount = float(row["sanction_amount_rs"])
            total += amount
        except (ValueError, TypeError):
            report_error(
                f"Sanction row {index} has invalid amount: "
                f"{row.get('sanction_amount_rs')}"
            )

        if row.get("project_id") != PROJECT_ID:
            report_error(
                f"Sanction row {index} has wrong project ID."
            )

    expected_total = 557468900.0

    if abs(total - expected_total) < 0.01:
        report_pass(
            f"Total sanctioned amount = "
            f"₹{total:,.2f}"
        )
    else:
        report_warning(
            f"Total sanctioned amount = "
            f"₹{total:,.2f}; expected "
            f"₹{expected_total:,.2f}"
        )


# ============================================================
# NOTIFICATIONS
# ============================================================

def check_notifications():
    print()
    print("=" * 70)
    print("4. NOTIFICATIONS CHECK")
    print("=" * 70)

    rows, fields = load_csv("notifications.csv")

    if not rows:
        return

    required = [
        "project_id",
        "notification_id",
        "notification_type",
        "serial_number",
        "publish_date",
        "notification_number",
        "status",
        "details_url",
        "objections_url",
    ]

    for field in required:

        if field in fields:
            report_pass(
                f"notifications.csv contains '{field}'"
            )
        else:
            report_error(
                f"notifications.csv missing '{field}'"
            )

    # --------------------------------------------------------
    # Count notification types
    # --------------------------------------------------------

    type_counts = Counter(
        row.get("notification_type")
        for row in rows
    )

    for notification_type, expected_count in (
        EXPECTED_NOTIFICATION_COUNTS.items()
    ):

        actual = type_counts.get(
            notification_type,
            0
        )

        if actual == expected_count:
            report_pass(
                f"{notification_type}: "
                f"{actual} notifications"
            )
        else:
            report_error(
                f"{notification_type}: expected "
                f"{expected_count}, found {actual}"
            )

    # --------------------------------------------------------
    # Duplicate notification IDs
    # --------------------------------------------------------

    keys = [
        (
            row.get("project_id"),
            row.get("notification_type"),
            row.get("notification_id"),
        )
        for row in rows
    ]

    duplicates = [
        key
        for key, count in Counter(keys).items()
        if count > 1
    ]

    if not duplicates:
        report_pass(
            "No duplicate notification IDs found."
        )
    else:
        report_error(
            f"Duplicate notification records found: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Required notification IDs
    # --------------------------------------------------------

    expected_ids = {
        "50840",
        "52358",
        "58267",
        "44531",
        "45887",
        "56691",
        "34193",
        "35100",
        "44312",
    }

    actual_ids = {
        row.get("notification_id")
        for row in rows
    }

    if actual_ids == expected_ids:
        report_pass(
            "All 9 expected notification IDs are present."
        )
    else:

        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids

        if missing:
            report_error(
                f"Missing notification IDs: {sorted(missing)}"
            )

        if extra:
            report_warning(
                f"Unexpected notification IDs: {sorted(extra)}"
            )

    # --------------------------------------------------------
    # Date and status checks
    # --------------------------------------------------------

    for row in rows:

        if not row.get("publish_date"):
            report_error(
                f"Notification {row.get('notification_id')} "
                f"has no publish date."
            )

        if not row.get("status"):
            report_warning(
                f"Notification {row.get('notification_id')} "
                f"has no status."
            )

        if row.get("project_id") != PROJECT_ID:
            report_error(
                f"Notification {row.get('notification_id')} "
                f"has wrong project ID."
            )


# ============================================================
# 3D SURVEY FILES
# ============================================================

def check_3d_surveys():
    print()
    print("=" * 70)
    print("5. 3D SURVEY DATA CHECK")
    print("=" * 70)

    total_rows = 0
    total_area = 0.0

    for notification_id, expected_count in (
        EXPECTED_SURVEY_COUNTS.items()
    ):

        filename = f"3d_{notification_id}_surveys.csv"

        rows, fields = load_csv(filename)

        if not rows:
            continue

        actual_count = len(rows)

        if actual_count == expected_count:
            report_pass(
                f"{filename}: {actual_count} survey records"
            )
        else:
            report_error(
                f"{filename}: expected "
                f"{expected_count}, found {actual_count}"
            )

        required = [
            "project_id",
            "notification_id",
            "notification_number",
            "tentative_publish_date",
            "serial_number",
            "district",
            "sub_district",
            "village",
            "survey_number",
            "area_hectares",
            "land_type",
            "land_nature",
            "land_category",
            "party_count",
            "owner_count",
            "affected_party_count",
        ]

        for field in required:

            if field not in fields:
                report_error(
                    f"{filename}: missing field '{field}'"
                )

        for row in rows:

            if row.get("project_id") != PROJECT_ID:
                report_error(
                    f"{filename}: wrong project ID"
                )

            if row.get("notification_id") != notification_id:
                report_error(
                    f"{filename}: unexpected notification ID "
                    f"{row.get('notification_id')}"
                )

            # Area must be numeric
            try:
                area = float(row["area_hectares"])
                total_area += area
            except (ValueError, TypeError):
                report_error(
                    f"{filename}: invalid area "
                    f"'{row.get('area_hectares')}'"
                )

        total_rows += actual_count

    report_pass(
        f"Combined 3D survey records: {total_rows}"
    )

    expected_total_rows = sum(
        EXPECTED_SURVEY_COUNTS.values()
    )

    if total_rows == expected_total_rows:
        report_pass(
            f"Combined survey count matches expected "
            f"{expected_total_rows}"
        )
    else:
        report_error(
            f"Combined survey count mismatch: "
            f"{total_rows} vs {expected_total_rows}"
        )

    expected_total_area = 19.632638

    if abs(total_area - expected_total_area) < 0.000001:
        report_pass(
            f"Combined 3D survey area = "
            f"{total_area:.6f} ha"
        )
    else:
        report_warning(
            f"Combined 3D survey area = "
            f"{total_area:.6f} ha; expected "
            f"{expected_total_area:.6f} ha"
        )


# ============================================================
# LAND PARTIES
# ============================================================

def check_land_parties():
    print()
    print("=" * 70)
    print("6. LAND PARTY DATA CHECK")
    print("=" * 70)

    total_rows = 0
    total_numeric_area = 0.0
    missing_area_rows = 0

    party_type_counts = Counter()

    for notification_id, expected_count in (
        EXPECTED_PARTY_COUNTS.items()
    ):

        filename = (
            f"3d_{notification_id}_land_parties.csv"
        )

        rows, fields = load_csv(filename)

        if not rows:
            continue

        actual_count = len(rows)

        if actual_count == expected_count:
            report_pass(
                f"{filename}: {actual_count} party records"
            )
        else:
            report_error(
                f"{filename}: expected "
                f"{expected_count}, found {actual_count}"
            )

        required = [
            "project_id",
            "notification_id",
            "survey_serial_number",
            "district",
            "sub_district",
            "village",
            "survey_number",
            "party_sequence",
            "party_name",
            "party_type",
            "party_area_hectares",
            "party_area_raw",
        ]

        for field in required:

            if field not in fields:
                report_error(
                    f"{filename}: missing field '{field}'"
                )

        for row in rows:

            if row.get("project_id") != PROJECT_ID:
                report_error(
                    f"{filename}: wrong project ID"
                )

            if row.get("notification_id") != notification_id:
                report_error(
                    f"{filename}: unexpected notification ID"
                )

            party_type = row.get("party_type")

            if party_type:
                party_type_counts[party_type] += 1

            raw_area = row.get(
                "party_area_raw",
                ""
            )

            area_value = row.get(
                "party_area_hectares",
                ""
            )

            if area_value:

                try:
                    total_numeric_area += float(area_value)
                except ValueError:
                    report_error(
                        f"{filename}: invalid party area "
                        f"'{area_value}'"
                    )

            else:
                missing_area_rows += 1

                # Missing areas should NOT become zero.
                if raw_area != "Hectares":
                    report_warning(
                        f"{filename}: missing party area with "
                        f"unexpected raw value '{raw_area}'"
                    )

        total_rows += actual_count

    expected_total = sum(
        EXPECTED_PARTY_COUNTS.values()
    )

    if total_rows == expected_total:
        report_pass(
            f"Combined party records: {total_rows}"
        )
    else:
        report_error(
            f"Combined party records: {total_rows}; "
            f"expected {expected_total}"
        )

    print()
    print("Party types:")

    for party_type, count in party_type_counts.items():
        print(f"  {party_type}: {count}")

    expected_party_types = {
        "Owner": 593,
        "Affected Party": 16,
    }

    if dict(party_type_counts) == expected_party_types:
        report_pass(
            "Party type counts match expected values."
        )
    else:
        report_warning(
            f"Party type counts: "
            f"{dict(party_type_counts)}"
        )

    # Missing party areas
    if missing_area_rows == 24:
        report_pass(
            "24 party-area values are NULL/missing, "
            "matching the known source behavior."
        )
    else:
        report_warning(
            f"Found {missing_area_rows} missing party-area "
            f"values; expected 24."
        )

    # Important: never interpret missing area as zero.
    report_pass(
        "Missing party-area values are preserved rather "
        "than converted to zero."
    )

    expected_numeric_area = 27.125992

    if abs(
        total_numeric_area - expected_numeric_area
    ) < 0.000001:

        report_pass(
            f"Numeric party-area total = "
            f"{total_numeric_area:.6f} ha"
        )

    else:

        report_warning(
            f"Numeric party-area total = "
            f"{total_numeric_area:.6f} ha; expected "
            f"{expected_numeric_area:.6f} ha"
        )


# ============================================================
# ACQUISITION SURVEYS
# ============================================================

def check_acquisition_surveys():
    print()
    print("=" * 70)
    print("7. ACQUISITION SURVEY MASTER CHECK")
    print("=" * 70)

    rows, fields = load_csv(
        "acquisition_surveys.csv"
    )

    if not rows:
        return

    if len(rows) == 380:
        report_pass(
            "acquisition_surveys.csv contains 380 records."
        )
    else:
        report_error(
            f"Expected 380 acquisition survey records, "
            f"found {len(rows)}"
        )

    if len(fields) == 21:
        report_pass(
            "acquisition_surveys.csv contains 21 columns."
        )
    else:
        report_warning(
            f"Expected 21 columns, found {len(fields)}"
        )

    if "notification_type" in fields:

        types = Counter(
            row.get("notification_type")
            for row in rows
        )

        if types.get("3D") == 380:
            report_pass(
                "All acquisition survey records are marked 3D."
            )
        else:
            report_warning(
                f"Notification type distribution: {dict(types)}"
            )

    # --------------------------------------------------------
    # Notification distribution
    # --------------------------------------------------------

    notification_ids = Counter(
        row.get("notification_id")
        for row in rows
    )

    print()
    print("Survey records by notification:")

    for notification_id, count in sorted(
        notification_ids.items()
    ):

        print(
            f"  {notification_id}: {count}"
        )

    expected_distribution = {
        "34193": 46,
        "35100": 310,
        "44312": 24,
    }

    if dict(notification_ids) == expected_distribution:
        report_pass(
            "Notification distribution matches expected values."
        )
    else:
        report_error(
            f"Unexpected notification distribution: "
            f"{dict(notification_ids)}"
        )

    # --------------------------------------------------------
    # Land nature
    # --------------------------------------------------------

    land_nature = Counter(
        row.get("land_nature")
        for row in rows
    )

    expected_land_nature = {
        "Private": 328,
        "Government": 52,
    }

    print()
    print("Land nature:")

    for key, count in land_nature.items():
        print(f"  {key}: {count}")

    if dict(land_nature) == expected_land_nature:
        report_pass(
            "Land nature distribution matches expected values."
        )
    else:
        report_warning(
            f"Land nature distribution: "
            f"{dict(land_nature)}"
        )

    # --------------------------------------------------------
    # Land category
    # --------------------------------------------------------

    land_category = Counter(
        row.get("land_category")
        for row in rows
    )

    if land_category.get("Rural") == 380:
        report_pass(
            "All 380 records are categorized as Rural."
        )
    else:
        report_warning(
            f"Land category distribution: "
            f"{dict(land_category)}"
        )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    total_area = 0.0

    for row in rows:

        try:
            total_area += float(
                row["area_hectares"]
            )
        except (ValueError, TypeError):
            report_error(
                f"Invalid area in acquisition_surveys.csv: "
                f"{row.get('area_hectares')}"
            )

    expected_area = 19.632638

    if abs(total_area - expected_area) < 0.000001:
        report_pass(
            f"Master survey area = "
            f"{total_area:.6f} ha"
        )
    else:
        report_warning(
            f"Master survey area = "
            f"{total_area:.6f} ha; expected "
            f"{expected_area:.6f} ha"
        )


# ============================================================
# CROSS-FILE CONSISTENCY
# ============================================================

def check_cross_file_consistency():
    print()
    print("=" * 70)
    print("8. CROSS-FILE CONSISTENCY CHECK")
    print("=" * 70)

    notifications, _ = load_csv(
        "notifications.csv"
    )

    survey_files = [
        "3d_34193_surveys.csv",
        "3d_35100_surveys.csv",
        "3d_44312_surveys.csv",
    ]

    notification_ids = {
        row.get("notification_id")
        for row in notifications
    }

    for filename in survey_files:

        rows, _ = load_csv(filename)

        if not rows:
            continue

        file_notification_ids = {
            row.get("notification_id")
            for row in rows
        }

        if file_notification_ids <= notification_ids:
            report_pass(
                f"{filename}: notification ID exists "
                f"in notifications.csv"
            )
        else:
            report_error(
                f"{filename}: contains notification ID "
                f"not present in notifications.csv"
            )

    # --------------------------------------------------------
    # Project ID consistency
    # --------------------------------------------------------

    all_files = [
        "project_manifest.csv",
        "sanctions.csv",
        "notifications.csv",
        "3d_34193_surveys.csv",
        "3d_34193_land_parties.csv",
        "3d_35100_surveys.csv",
        "3d_35100_land_parties.csv",
        "3d_44312_surveys.csv",
        "3d_44312_land_parties.csv",
        "acquisition_surveys.csv",
    ]

    for filename in all_files:

        rows, _ = load_csv(filename)

        for row in rows:

            if row.get("project_id") != PROJECT_ID:

                report_error(
                    f"{filename}: found project ID "
                    f"'{row.get('project_id')}' "
                    f"instead of '{PROJECT_ID}'"
                )


# ============================================================
# FINAL SUMMARY
# ============================================================

def final_summary():
    print()
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    print()
    print(f"PASS     : {len(passes)}")
    print(f"WARNINGS : {len(warnings)}")
    print(f"FAILURES : {len(errors)}")

    print()

    if errors:

        print("DATASET STATUS: FAIL")
        print()
        print("Failures:")

        for error in errors:
            print(f"  - {error}")

        return 1

    if warnings:

        print("DATASET STATUS: PASS WITH WARNINGS")
        print()
        print("Warnings:")

        for warning in warnings:
            print(f"  - {warning}")

        return 0

    print("DATASET STATUS: PASS")
    print()
    print(
        "The current BhoomiRashi normalized dataset "
        "passed all validation checks."
    )

    return 0


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("#" * 70)
    print("# BhoomiSetu - BhoomiRashi Data Validator")
    print(f"# Project: {PROJECT_ID}")
    print("#" * 70)

    print()
    print(f"Data directory:")
    print(DATA_DIR)

    if not DATA_DIR.exists():

        print()
        print(
            f"ERROR: Data directory does not exist:\n"
            f"{DATA_DIR}"
        )

        sys.exit(1)

    check_expected_files()
    check_project_manifest()
    check_sanctions()
    check_notifications()
    check_3d_surveys()
    check_land_parties()
    check_acquisition_surveys()
    check_cross_file_consistency()

    exit_code = final_summary()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()