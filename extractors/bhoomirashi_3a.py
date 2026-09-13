import argparse
import csv
import re
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_PROJECT_ID = "60432"
DEFAULT_NOTIFICATION_ID = "59173"

BASE_RAW_DIR = (
    Path("output")
    / "raw"
    / "bhoomirashi"
)

BASE_OUTPUT_DIR = (
    Path("output")
    / "normalized"
    / "bhoomirashi"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(value):
    """
    Clean whitespace while preserving actual text.
    """

    if value is None:
        return ""

    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def parse_date(value):
    """
    Convert DD/MM/YYYY to YYYY-MM-DD.
    """

    value = clean_text(value)

    if not value:
        return ""

    try:
        dt = datetime.strptime(
            value,
            "%d/%m/%Y"
        )

        return dt.strftime("%Y-%m-%d")

    except ValueError:
        return value


# ============================================================
# ACCESS / LOGIN PAGE DETECTION
# ============================================================

def detect_access_restriction(soup, html):
    """
    Detect whether BhoomiRashi returned a login,
    maintenance, redirect, or generic portal page instead
    of the requested 3a notification detail page.

    This is important because some BhoomiRashi notification
    detail URLs currently redirect to login1.cshtml.

    Returns:
        True  -> access restricted / wrong page
        False -> page appears to contain actual content
    """

    page_text = clean_text(
        soup.get_text(
            " ",
            strip=True
        )
    ).lower()

    html_lower = html.lower()

    # --------------------------------------------------------
    # Login / authentication indicators
    # --------------------------------------------------------

    login_indicators = [
        "login1.cshtml",
        "username",
        "password",
        "captcha",
        "recaptcha",
        "forgot password",
        "sign in",
    ]

    login_hits = sum(
        1
        for indicator in login_indicators
        if indicator in page_text
        or indicator in html_lower
    )

    # --------------------------------------------------------
    # Maintenance / generic portal indicators
    # --------------------------------------------------------

    maintenance_indicators = [
        "site is under maintenance",
        "do not login",
        "land acquisition portal",
    ]

    maintenance_hits = sum(
        1
        for indicator in maintenance_indicators
        if indicator in page_text
        or indicator in html_lower
    )

    # --------------------------------------------------------
    # Actual 3a page indicators
    # --------------------------------------------------------

    actual_3a_indicators = [
        "details 3a notification",
        "tentative publish date",
        "file number",
    ]

    actual_3a_hits = sum(
        1
        for indicator in actual_3a_indicators
        if indicator in page_text
        or indicator in html_lower
    )

    # --------------------------------------------------------
    # Strong evidence of restricted page
    #
    # A login/captcha page without actual 3a content should
    # be treated as inaccessible.
    # --------------------------------------------------------

    if actual_3a_hits == 0:

        if login_hits >= 2:
            return True

        if maintenance_hits >= 1 and login_hits >= 1:
            return True

    return False


# ============================================================
# NOTIFICATION METADATA
# ============================================================

def extract_notification_metadata(soup):
    """
    Extract metadata from the 3a notification page.

    Expected fields:

        Notification Number
        Tentative Publish Date
        File Number
        CALA
    """

    tables = soup.find_all("table")

    metadata = {
        "notification_number": "",
        "tentative_publish_date": "",
        "file_number": "",
        "cala_name": "",
    }

    if not tables:
        return metadata

    # --------------------------------------------------------
    # Extract label/value pairs
    # --------------------------------------------------------

    for table in tables:

        rows = table.find_all("tr")

        for row in rows:

            cells = row.find_all(
                ["td", "th"],
                recursive=False
            )

            values = [
                clean_text(
                    cell.get_text(
                        " ",
                        strip=True
                    )
                )
                for cell in cells
            ]

            if len(values) < 2:
                continue

            label = values[0]
            value = values[1]

            if label == "Tentative Publish Date":

                metadata[
                    "tentative_publish_date"
                ] = parse_date(value)

            elif label == "File Number":

                metadata[
                    "file_number"
                ] = value

    # --------------------------------------------------------
    # Extract notification number
    #
    # Observed page:
    #
    # Details 3a Notification for "4459
    #
    # We intentionally stop at whitespace or quote so that
    # the rest of the page text is not captured.
    # --------------------------------------------------------

    page_text = clean_text(
        soup.get_text(
            " ",
            strip=True
        )
    )

    match = re.search(
        r'Details\s+3a\s+Notification\s+for\s*"([^\s"]+)',
        page_text,
        flags=re.IGNORECASE
    )

    if match:

        metadata[
            "notification_number"
        ] = clean_text(
            match.group(1)
        )

    # --------------------------------------------------------
    # Fallback: inspect raw HTML text
    # --------------------------------------------------------

    if not metadata["notification_number"]:

        raw_text = str(soup)

        match = re.search(
            r'Details\s+3a\s+Notification\s+for\s*[\'"]([^\s\'"]+)',
            raw_text,
            flags=re.IGNORECASE
        )

        if match:

            metadata[
                "notification_number"
            ] = clean_text(
                match.group(1)
            )

    # --------------------------------------------------------
    # CALA
    # --------------------------------------------------------

    for table in tables:

        rows = table.find_all("tr")

        for row in rows:

            cells = row.find_all(
                ["td", "th"],
                recursive=False
            )

            values = [
                clean_text(
                    cell.get_text(
                        " ",
                        strip=True
                    )
                )
                for cell in cells
            ]

            for value in values:

                if "CALA" not in value.upper():
                    continue

                # ------------------------------------------------
                # The page contains English followed by Hindi.
                #
                # Keep the English representation.
                # ------------------------------------------------

                english_part = re.split(
                    r"\s+[अ-हक़-य़]",
                    value,
                    maxsplit=1
                )[0]

                english_part = clean_text(
                    english_part
                )

                if english_part:

                    metadata[
                        "cala_name"
                    ] = english_part

                    break

            if metadata["cala_name"]:
                break

        if metadata["cala_name"]:
            break

    return metadata


# ============================================================
# VILLAGE EXTRACTION
# ============================================================

def extract_villages(soup):
    """
    Extract village/action records.

    The 3a page contains:

        - metadata tables
        - nested village tables
        - duplicated village rows

    Only genuine village/action rows are accepted.
    """

    villages = []

    seen = set()

    # --------------------------------------------------------
    # Known metadata/header values
    # --------------------------------------------------------

    invalid_first_values = {
        "",
        "villages",
        "tentative publish date",
        "file number",
    }

    # --------------------------------------------------------
    # Known/expected BhoomiRashi village actions
    # --------------------------------------------------------

    valid_actions = {
        "added",
        "deleted",
        "modified",
        "updated",
        "removed",
    }

    for table in soup.find_all("table"):

        rows = table.find_all("tr")

        for row in rows:

            cells = row.find_all(
                ["td", "th"],
                recursive=False
            )

            # Genuine village/action row:
            # exactly two direct cells.
            if len(cells) != 2:
                continue

            values = [
                clean_text(
                    cell.get_text(
                        " ",
                        strip=True
                    )
                )
                for cell in cells
            ]

            village = values[0]
            action = values[1]

            # ------------------------------------------------
            # Ignore headers and metadata
            # ------------------------------------------------

            if village.lower() in invalid_first_values:
                continue

            if action.lower() == "action":
                continue

            if not village or not action:
                continue

            # ------------------------------------------------
            # Ignore long metadata/CALA rows
            # ------------------------------------------------

            if len(action) > 50:
                continue

            if len(village) > 100:
                continue

            # ------------------------------------------------
            # Accept only known action values
            # ------------------------------------------------

            if action.lower() not in valid_actions:
                continue

            # ------------------------------------------------
            # Deduplicate nested tables
            # ------------------------------------------------

            unique_key = (
                village.lower(),
                action.lower(),
            )

            if unique_key in seen:
                continue

            seen.add(unique_key)

            villages.append({
                "village": village,
                "action": action,
            })

    return villages


# ============================================================
# CSV WRITERS
# ============================================================

def save_metadata_csv(
    metadata,
    project_id,
    notification_id,
    output_dir
):
    """
    Save 3a notification metadata.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"3a_{notification_id}_metadata.csv"
    )

    fieldnames = [
        "project_id",
        "notification_id",
        "notification_number",
        "tentative_publish_date",
        "file_number",
        "cala_name",
    ]

    row = {
        "project_id":
            project_id,

        "notification_id":
            notification_id,

        "notification_number":
            metadata.get(
                "notification_number",
                ""
            ),

        "tentative_publish_date":
            metadata.get(
                "tentative_publish_date",
                ""
            ),

        "file_number":
            metadata.get(
                "file_number",
                ""
            ),

        "cala_name":
            metadata.get(
                "cala_name",
                ""
            ),
    }

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerow(row)

    return output_file


def save_villages_csv(
    villages,
    project_id,
    notification_id,
    metadata,
    output_dir
):
    """
    Save 3a village/action records.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / f"3a_{notification_id}_villages.csv"
    )

    fieldnames = [
        "project_id",
        "notification_id",
        "notification_number",
        "tentative_publish_date",
        "file_number",
        "cala_name",
        "village",
        "action",
    ]

    rows = []

    for village in villages:

        rows.append({

            "project_id":
                project_id,

            "notification_id":
                notification_id,

            "notification_number":
                metadata.get(
                    "notification_number",
                    ""
                ),

            "tentative_publish_date":
                metadata.get(
                    "tentative_publish_date",
                    ""
                ),

            "file_number":
                metadata.get(
                    "file_number",
                    ""
                ),

            "cala_name":
                metadata.get(
                    "cala_name",
                    ""
                ),

            "village":
                village["village"],

            "action":
                village["action"],
        })

    with output_file.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    return output_file


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Extract BhoomiRashi "
            "3a notification data"
        )
    )

    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=(
            "BhoomiRashi numeric "
            "project ID"
        )
    )

    parser.add_argument(
        "--notification-id",
        default=DEFAULT_NOTIFICATION_ID,
        help=(
            "BhoomiRashi 3a "
            "notification ID"
        )
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    project_id = str(
        args.project_id
    ).strip()

    notification_id = str(
        args.notification_id
    ).strip()

    source_html = (
        BASE_RAW_DIR
        / project_id
        / f"3a_{notification_id}_test.html"
    )

    output_dir = (
        BASE_OUTPUT_DIR
        / project_id
    )

    print("=" * 70)
    print(
        "BHOOMIRASHI 3a NOTIFICATION EXTRACTOR"
    )
    print("=" * 70)

    print(
        f"Project ID       : {project_id}"
    )

    print(
        f"Notification ID  : {notification_id}"
    )

    print(
        f"Source            : {source_html}"
    )

    print(
        f"Output            : {output_dir}"
    )

    print()

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not source_html.exists():

        print(
            "ERROR: Source HTML not found:"
        )

        print(source_html)

        print()

        print(
            "Expected file:"
        )

        print(
            f"output/raw/bhoomirashi/"
            f"{project_id}/"
            f"3a_{notification_id}_test.html"
        )

        return 1

    # --------------------------------------------------------
    # Read HTML
    # --------------------------------------------------------

    html = source_html.read_text(
        encoding="utf-8"
    )

    soup = BeautifulSoup(
        html,
        "lxml"
    )

    print(
        f"Loaded HTML "
        f"({len(html):,} characters)"
    )

    print(
        f"Found "
        f"{len(soup.find_all('table'))} tables"
    )

    print()

    # --------------------------------------------------------
    # Detect login / maintenance / access restriction
    # --------------------------------------------------------

    if detect_access_restriction(
        soup,
        html
    ):

        print("=" * 70)
        print(
            "ACCESS RESTRICTED"
        )
        print("=" * 70)

        print()

        print(
            "The downloaded HTML appears to be "
            "a BhoomiRashi login/maintenance/"
            "generic portal page rather than the "
            "requested 3a notification detail page."
        )

        print()

        print(
            f"Project ID      : {project_id}"
        )

        print(
            f"Notification ID : {notification_id}"
        )

        print()

        print(
            "The 3a detail data could not be "
            "extracted."
        )

        print(
            "No empty CSV files were created."
        )

        print()

        print(
            "This should be treated as "
            "ACCESS_RESTRICTED, not as "
            "zero village records."
        )

        print()

        print("=" * 70)

        return 2

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = (
        extract_notification_metadata(
            soup
        )
    )

    print(
        "NOTIFICATION METADATA"
    )

    print("-" * 70)

    print(
        f"Notification Number : "
        f"{metadata['notification_number']}"
    )

    print(
        f"Tentative Publish Date : "
        f"{metadata['tentative_publish_date']}"
    )

    print(
        f"File Number : "
        f"{metadata['file_number']}"
    )

    print(
        f"CALA : "
        f"{metadata['cala_name']}"
    )

    print()

    # --------------------------------------------------------
    # Villages
    # --------------------------------------------------------

    villages = extract_villages(
        soup
    )

    print(
        "VILLAGES"
    )

    print("-" * 70)

    for village in villages:

        print(
            f"{village['village']} | "
            f"{village['action']}"
        )

    print()

    print(
        f"Village records : "
        f"{len(villages)}"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata_file = (
        save_metadata_csv(
            metadata,
            project_id,
            notification_id,
            output_dir
        )
    )

    # --------------------------------------------------------
    # Save villages
    # --------------------------------------------------------

    villages_file = (
        save_villages_csv(
            villages,
            project_id,
            notification_id,
            metadata,
            output_dir
        )
    )

    print()

    print(
        "FILES SAVED"
    )

    print("-" * 70)

    print(metadata_file)

    print(villages_file)

    print()

    print("=" * 70)

    print(
        "Extraction completed successfully."
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