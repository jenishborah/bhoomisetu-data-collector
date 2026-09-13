import argparse
import csv
import re
from pathlib import Path
from bs4 import BeautifulSoup
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_PROJECT_ID = "54635"

BASE_URL = "https://bhoomirashi.gov.in/auth/revamp/"

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
    Clean whitespace while preserving the actual text.
    """

    if value is None:
        return ""

    value = value.replace("\xa0", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def parse_hectares(value):
    """
    Extract numeric hectare value from strings such as:

        22 (ha)
        0.5415 (ha)
        22 ha
    """

    value = clean_text(value)

    if not value:
        return None

    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)",
        value
    )

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


def parse_amount(value):
    """
    Convert an amount string into float.
    """

    value = clean_text(value)

    if not value:
        return None

    value = value.replace(",", "")

    match = re.search(
        r"[-+]?\d+(?:\.\d+)?",
        value
    )

    if not match:
        return None

    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_date(value):
    """
    Validate/normalize date in DD/MM/YYYY format.
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
# PROJECT DETAILS
# ============================================================

def extract_project_details(soup):
    """
    Extract basic project details from the first table.
    """

    tables = soup.find_all("table")

    if not tables:
        return {}

    table = tables[0]

    values = [
        clean_text(
            td.get_text(" ", strip=True)
        )
        for td in table.find_all(
            ["td", "th"]
        )
    ]

    details = {}

    for i, value in enumerate(values):

        if (
            value == "Project Name"
            and i + 1 < len(values)
        ):

            details["project_name"] = (
                values[i + 1]
            )

        elif (
            value == "Project Number"
            and i + 1 < len(values)
        ):

            details["project_number"] = (
                values[i + 1]
            )

        elif (
            value == "Land Required"
            and i + 1 < len(values)
        ):

            details["land_required_ha"] = (
                parse_hectares(values[i + 1])
            )

        elif (
            value == "Land Available"
            and i + 1 < len(values)
        ):

            details["land_available_ha"] = (
                parse_hectares(values[i + 1])
            )

        elif (
            value == "Land to be acquired"
            and i + 1 < len(values)
        ):

            details["land_to_acquire_ha"] = (
                parse_hectares(values[i + 1])
            )

        elif (
            value == "Land Acquired till now"
            and i + 1 < len(values)
        ):

            details["land_acquired_ha"] = (
                parse_hectares(values[i + 1])
            )

    return details


# ============================================================
# SANCTIONS
# ============================================================

def extract_sanctions(soup, project_id):
    """
    Extract project sanction records.

    Expected structure:

    Project Sanction Details
    Sanction Number | Sanction Date |
    Sanction Amount (Rs.) | Total (Rs.)
    """

    tables = soup.find_all("table")

    if len(tables) < 2:
        return []

    table = tables[1]

    rows = table.find_all("tr")

    sanctions = []

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

        if len(values) < 3:
            continue

        sanction_number = values[0]
        sanction_date = values[1]
        sanction_amount = values[2]

        # Skip title/header/total rows
        if (
            not sanction_number
            or sanction_number == "Sanction Number"
            or sanction_number.lower() == "total"
        ):
            continue

        amount = parse_amount(
            sanction_amount
        )

        if amount is None:
            continue

        sanctions.append({

            "project_id": project_id,

            "sanction_number":
                sanction_number,

            "sanction_date":
                parse_date(
                    sanction_date
                ),

            "sanction_amount_rs":
                amount,
        })

    return sanctions


# ============================================================
# NOTIFICATION HELPERS
# ============================================================

def extract_query_parameter(
    href,
    parameter
):
    """
    Extract a query parameter from a URL.

    Example:
        notification_id=50840
    """

    if not href:
        return ""

    pattern = (
        rf"(?:[?&])"
        rf"{re.escape(parameter)}"
        rf"=([^&]+)"
    )

    match = re.search(
        pattern,
        href
    )

    if not match:
        return ""

    return match.group(1).strip()


def classify_notification_table(
    table_index
):
    """
    Map project report table indexes
    to notification types.

    Observed BhoomiRashi structure:

        Table 2 -> 3a
        Table 3 -> 3A
        Table 4 -> 3D
    """

    mapping = {

        2: "3a",

        3: "3A",

        4: "3D",
    }

    return mapping.get(
        table_index
    )


def extract_notifications(
    soup,
    project_id
):
    """
    Extract project-level 3a, 3A and 3D
    notifications.

    BhoomiRashi displays bilingual rows twice.

    Deduplication therefore uses:

        project_id
        notification_type
        notification_id

    """

    tables = soup.find_all("table")

    notifications = []

    seen = set()

    for table_index in [2, 3, 4]:

        if table_index >= len(tables):
            continue

        table = tables[
            table_index
        ]

        notification_type = (
            classify_notification_table(
                table_index
            )
        )

        if not notification_type:
            continue

        rows = table.find_all("tr")

        for row in rows:

            cells = row.find_all(
                ["td", "th"],
                recursive=False
            )

            if not cells:
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

            if not values:
                continue

            # Skip header
            if values[0] in [
                "Sno",
                "S.No"
            ]:
                continue

            if len(values) < 4:
                continue

            serial_number = values[0]

            publish_date = values[1]

            notification_number_raw = (
                values[2]
            )

            notification_number = (
                clean_notification_number(
                    notification_number_raw
                )
            )

            # ------------------------------------------------
            # Locate links
            # ------------------------------------------------

            links = row.find_all("a")

            details_url = ""

            objections_url = ""

            for link in links:

                href = link.get(
                    "href",
                    ""
                )

                link_text = clean_text(
                    link.get_text(
                        " ",
                        strip=True
                    )
                ).lower()

                if "view details" in link_text:

                    if href:
                        details_url = href

                elif "view objections" in link_text:

                    if href:
                        objections_url = href

            # ------------------------------------------------
            # Extract notification ID
            # ------------------------------------------------

            notification_id = (
                extract_query_parameter(
                    details_url,
                    "notification_id"
                )
            )

            if not notification_id:

                notification_id = (
                    extract_query_parameter(
                        objections_url,
                        "notification_id"
                    )
                )

            if not notification_id:
                continue

            # ------------------------------------------------
            # Deduplicate bilingual rows
            # ------------------------------------------------

            unique_key = (

                project_id,

                notification_type,

                notification_id,
            )

            if unique_key in seen:
                continue

            seen.add(unique_key)

            # ------------------------------------------------
            # Normalize URLs
            # ------------------------------------------------

            details_url = normalize_url(
                details_url
            )

            objections_url = normalize_url(
                objections_url
            )

            # ------------------------------------------------
            # Store record
            # ------------------------------------------------

            notifications.append({

                "project_id":
                    project_id,

                "notification_id":
                    notification_id,

                "notification_type":
                    notification_type,

                "serial_number":
                    serial_number,

                "publish_date":
                    parse_date(
                        publish_date
                    ),

                "notification_number":
                    notification_number,

                "status":
                    values[-1],

                "details_url":
                    details_url,

                "objections_url":
                    objections_url,
            })

    return notifications


def clean_notification_number(
    value
):
    """
    Clean bilingual notification numbers.

    Examples:

        1405 १४०५
            -> 1405

        2685 २६८५
            -> 2685

        3761(E)
            -> 3761(E)

        S.O. 5048 (E)
        S.O. ५०४८ (E)
            -> S.O. 5048 (E)
    """

    value = clean_text(value)

    if not value:
        return ""

    # --------------------------------------------------------
    # S.O. bilingual format
    # --------------------------------------------------------

    match = re.match(

        r"^(S\.O\.\s+\d+\s*\([A-Za-z]\))"
        r"\s+S\.O\.\s+[०-९]+\s*\([A-Za-z]\)$",

        value,

        flags=re.IGNORECASE,
    )

    if match:

        return clean_text(
            match.group(1)
        )

    # --------------------------------------------------------
    # Simple bilingual duplicate
    # --------------------------------------------------------

    match = re.match(

        r"^([A-Za-z0-9()./\- ]+?)"
        r"\s+([०-९]+)$",

        value,
    )

    if match:

        return clean_text(
            match.group(1)
        )

    # --------------------------------------------------------
    # Generic duplicated representation
    # --------------------------------------------------------

    parts = value.split()

    if len(parts) % 2 == 0:

        midpoint = (
            len(parts) // 2
        )

        first = " ".join(
            parts[:midpoint]
        )

        second = " ".join(
            parts[midpoint:]
        )

        if (
            normalize_for_comparison(
                first
            )
            ==
            normalize_for_comparison(
                second
            )
        ):

            return first

    return value


def normalize_for_comparison(
    value
):
    """
    Normalize strings for bilingual
    duplicate comparison.
    """

    value = clean_text(
        value
    )

    value = re.sub(
        r"[०-९]",
        "",
        value
    )

    value = re.sub(
        r"\s+",
        "",
        value
    )

    return value.lower()


def normalize_url(href):
    """
    Convert BhoomiRashi links into a
    stable relative path representation.

    We do not invent endpoints.
    """

    if not href:
        return ""

    href = href.strip()

    # Javascript wrapper
    match = re.search(

        r"""['"]([^'"]+\.cshtml[^'"]*)['"]""",

        href,

        flags=re.IGNORECASE
    )

    if match:

        href = match.group(1)

    href = href.replace(
        "\\'",
        "'"
    )

    href = href.rstrip(
        "'\" )"
    )

    return href


# ============================================================
# CSV WRITERS
# ============================================================

def save_project_csv(
    project_details,
    project_id,
    output_dir
):
    """
    Save project_manifest.csv
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "project_manifest.csv"
    )

    fieldnames = [

        "project_id",

        "project_name",

        "project_number",

        "land_required_ha",

        "land_available_ha",

        "land_to_acquire_ha",

        "land_acquired_ha",
    ]

    row = {

        "project_id":
            project_id,

        "project_name":
            project_details.get(
                "project_name",
                ""
            ),

        "project_number":
            project_details.get(
                "project_number",
                ""
            ),

        "land_required_ha":
            project_details.get(
                "land_required_ha"
            ),

        "land_available_ha":
            project_details.get(
                "land_available_ha"
            ),

        "land_to_acquire_ha":
            project_details.get(
                "land_to_acquire_ha"
            ),

        "land_acquired_ha":
            project_details.get(
                "land_acquired_ha"
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


def save_sanctions_csv(
    sanctions,
    output_dir
):
    """
    Save sanctions.csv
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "sanctions.csv"
    )

    fieldnames = [

        "project_id",

        "sanction_number",

        "sanction_date",

        "sanction_amount_rs",
    ]

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

        writer.writerows(
            sanctions
        )

    return output_file


def save_notifications_csv(
    notifications,
    output_dir
):
    """
    Save normalized notification timeline.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_dir
        / "notifications.csv"
    )

    fieldnames = [

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

        writer.writerows(
            notifications
        )

    return output_file


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(

        description=(
            "Extract BhoomiRashi "
            "project-level data"
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

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    project_id = str(
        args.project_id
    ).strip()

    source_html = (
        BASE_RAW_DIR
        / project_id
        / "source.html"
    )

    output_dir = (
        BASE_OUTPUT_DIR
        / project_id
    )

    print("=" * 70)
    print("BhoomiRashi Project Extractor")
    print("=" * 70)

    print(
        f"Project ID : {project_id}"
    )

    print(
        f"Source     : {source_html}"
    )

    print(
        f"Output     : {output_dir}"
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
            "The project report HTML must "
            "be fetched first."
        )

        print(
            "Expected URL:"
        )

        print(
            f"{BASE_URL}"
            f"prep_public.cshtml?"
            f"nid=1&project_id={project_id}"
        )

        return

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
    # Project details
    # --------------------------------------------------------

    project_details = (
        extract_project_details(
            soup
        )
    )

    print(
        "PROJECT DETAILS"
    )

    print("-" * 70)

    for key, value in (
        project_details.items()
    ):

        print(
            f"{key:25}: {value}"
        )

    print()

    # --------------------------------------------------------
    # Sanctions
    # --------------------------------------------------------

    sanctions = (
        extract_sanctions(
            soup,
            project_id
        )
    )

    print(
        "SANCTIONS"
    )

    print("-" * 70)

    for sanction in sanctions:

        print(

            f"{sanction['sanction_number']} | "

            f"{sanction['sanction_date']} | "

            f"₹"
            f"{sanction['sanction_amount_rs']:,.2f}"
        )

    print(
        f"Total sanctions: "
        f"{len(sanctions)}"
    )

    total_sanction_amount = sum(

        s["sanction_amount_rs"]

        for s in sanctions

        if s["sanction_amount_rs"]
        is not None
    )

    print(

        f"Total sanctioned amount: "
        f"₹"
        f"{total_sanction_amount:,.2f}"
    )

    print()

    # --------------------------------------------------------
    # Notifications
    # --------------------------------------------------------

    notifications = (
        extract_notifications(
            soup,
            project_id
        )
    )

    print(
        "NOTIFICATIONS"
    )

    print("-" * 70)

    for notification in notifications:

        print(

            f"{notification['notification_type']:>2} | "

            f"ID "
            f"{notification['notification_id']} | "

            f"{notification['publish_date']} | "

            f"{notification['notification_number']} | "

            f"{notification['status']}"
        )

    print()

    # --------------------------------------------------------
    # Notification counts
    # --------------------------------------------------------

    notification_counts = {}

    for notification in notifications:

        notification_type = (
            notification[
                "notification_type"
            ]
        )

        notification_counts[
            notification_type
        ] = (

            notification_counts.get(
                notification_type,
                0
            )
            + 1
        )

    print(
        "NOTIFICATION COUNTS"
    )

    print("-" * 70)

    for notification_type in [
        "3a",
        "3A",
        "3D"
    ]:

        print(

            f"{notification_type}: "

            f"{notification_counts.get(
                notification_type,
                0
            )}"
        )

    print()

    # --------------------------------------------------------
    # Save files
    # --------------------------------------------------------

    project_file = (
        save_project_csv(
            project_details,
            project_id,
            output_dir
        )
    )

    sanctions_file = (
        save_sanctions_csv(
            sanctions,
            output_dir
        )
    )

    notifications_file = (
        save_notifications_csv(
            notifications,
            output_dir
        )
    )

    print(
        "FILES SAVED"
    )

    print("-" * 70)

    print(project_file)

    print(sanctions_file)

    print(notifications_file)

    print()

    print("=" * 70)

    print(
        "Extraction completed successfully."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()