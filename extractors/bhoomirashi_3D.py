import argparse
import csv
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = "https://bhoomirashi.gov.in/auth/revamp/"

DEFAULT_PROJECT_ID = "54635"
DEFAULT_NOTIFICATION_ID = "34193"

OUTPUT_DIR = (
    Path(__file__).resolve().parent.parent
    / "output"
    / "normalized"
    / "bhoomirashi"
)


# ============================================================
# HTTP
# ============================================================

def fetch_page(url: str) -> str:

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/153.0 Safari/537.36"
        ),
        "Accept": (
            "text/html,application/xhtml+xml,"
            "application/xml;q=0.9,*/*;q=0.8"
        ),
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.text


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_text(value) -> str:

    if value is None:
        return ""

    text = str(value)

    text = text.replace("\xa0", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_header(value: str) -> str:

    value = clean_text(value).lower()

    value = value.replace("\n", " ")
    value = value.replace(":", "")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def get_direct_cells(row):

    return row.find_all(
        ["td", "th"],
        recursive=False,
    )


# ============================================================
# AREA
# ============================================================

def parse_area(value):

    text = clean_text(value)

    if not text:
        return None

    match = re.search(
        r"([-+]?\d+(?:\.\d+)?)",
        text,
    )

    if not match:
        return None

    try:
        return float(match.group(1))
    except ValueError:
        return None


# ============================================================
# LAND TYPE
# ============================================================

def parse_land_type(value: str):

    text = clean_text(value)

    land_type = None
    land_nature = None
    land_category = None

    match = re.search(
        r'Land Type\s*:\s*"([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )

    if match:
        land_type = clean_text(
            match.group(1)
        )

    match = re.search(
        r'Land Nature\s*:\s*"([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )

    if match:
        land_nature = clean_text(
            match.group(1)
        )

    match = re.search(
        r'Land Category\s*:\s*"([^"]+)"',
        text,
        flags=re.IGNORECASE,
    )

    if match:
        land_category = clean_text(
            match.group(1)
        )

    return (
        land_type,
        land_nature,
        land_category,
    )


# ============================================================
# SURVEY NUMBER
# ============================================================

def clean_survey_number(value: str):

    text = clean_text(value)

    if not text:
        return ""

    parts = re.split(
        r"\s+",
        text,
    )

    return parts[0].strip()


# ============================================================
# NOTIFICATION METADATA
# ============================================================

def extract_notification_metadata(soup):

    tables = soup.find_all("table")

    if not tables:

        return {
            "notification_number": None,
            "tentative_publish_date": None,
        }

    wrapper_table = tables[0]

    text = clean_text(
        wrapper_table.get_text(
            " ",
            strip=True,
        )
    )

    notification_number = None

    title_match = re.search(
        r"Details\s+of\s+Survey\s+Numbers\s+for\s*"
        r"[\"“”]?\s*(.*?)\s*"
        r"(?=Tentative\s+Publish\s+Date)",
        text,
        flags=re.IGNORECASE,
    )

    if title_match:

        notification_number = clean_text(
            title_match.group(1)
        )

        notification_number = (
            notification_number
            .strip('"')
            .strip("“")
            .strip("”")
            .strip()
        )

    tentative_publish_date = None

    date_match = re.search(
        r"Tentative\s+Publish\s+Date\s+"
        r"(\d{2}/\d{2}/\d{4})",
        text,
        flags=re.IGNORECASE,
    )

    if date_match:

        tentative_publish_date = (
            date_match.group(1)
        )

    return {
        "notification_number": (
            notification_number
        ),
        "tentative_publish_date": (
            tentative_publish_date
        ),
    }


# ============================================================
# LAND PARTIES
# ============================================================

def extract_land_parties_from_party_row(
    party_row
):

    parties = []

    nested_tables = party_row.find_all(
        "table"
    )

    for table in nested_tables:

        rows = table.find_all("tr")

        if not rows:
            continue

        header_row = rows[0]

        headers = [
            normalize_header(
                cell.get_text(
                    " ",
                    strip=True,
                )
            )
            for cell in header_row.find_all(
                ["th", "td"],
                recursive=False,
            )
        ]

        required_headers = {
            "name",
            "address",
            "type",
            "area",
        }

        if not required_headers.issubset(
            set(headers)
        ):
            continue

        positions = {
            header: index
            for index, header in enumerate(
                headers
            )
        }

        for row in rows[1:]:

            cells = [
                clean_text(
                    cell.get_text(
                        " ",
                        strip=True,
                    )
                )
                for cell in row.find_all(
                    ["td", "th"],
                    recursive=False,
                )
            ]

            if len(cells) < 4:
                continue

            name = cells[
                positions["name"]
            ]

            address = cells[
                positions["address"]
            ]

            party_type = cells[
                positions["type"]
            ]

            area_raw = cells[
                positions["area"]
            ]

            if not any(
                [
                    name,
                    address,
                    party_type,
                    area_raw,
                ]
            ):
                continue

            parties.append(
                {
                    "party_name": (
                        name or None
                    ),
                    "party_address": (
                        address or None
                    ),
                    "party_type": (
                        party_type or None
                    ),
                    "party_area_hectares": (
                        parse_area(area_raw)
                    ),
                    "party_area_raw": (
                        area_raw or None
                    ),
                }
            )

    return parties


# ============================================================
# MAIN SURVEY TABLE
# ============================================================

def find_main_survey_table(soup):

    candidate_tables = soup.find_all(
        "table"
    )

    for table in candidate_tables:

        rows = table.find_all("tr")

        if not rows:
            continue

        for row in rows:

            headers = [
                normalize_header(
                    cell.get_text(
                        " ",
                        strip=True,
                    )
                )
                for cell in row.find_all(
                    ["th", "td"],
                    recursive=False,
                )
            ]

            header_set = set(headers)

            has_serial = (
                "s.no" in header_set
                or "s no" in header_set
            )

            has_district = (
                "district" in header_set
            )

            has_village = (
                "village" in header_set
            )

            has_survey = (
                "survey number"
                in header_set
                or "survey no."
                in header_set
                or "survey no"
                in header_set
            )

            has_area = (
                "area" in header_set
            )

            if (
                has_serial
                and has_district
                and has_village
                and has_survey
                and has_area
            ):

                return table

    return None


# ============================================================
# HEADER POSITIONS
# ============================================================

def get_header_positions(table):

    rows = table.find_all("tr")

    for row in rows:

        cells = row.find_all(
            ["th", "td"],
            recursive=False,
        )

        headers = [
            normalize_header(
                cell.get_text(
                    " ",
                    strip=True,
                )
            )
            for cell in cells
        ]

        positions = {}

        for index, header in enumerate(
            headers
        ):

            if header in {
                "s.no",
                "s no",
            }:

                positions[
                    "serial_number"
                ] = index

            elif header == "district":

                positions[
                    "district"
                ] = index

            elif header in {
                "sub district",
                "sub-district",
                "subdistrict",
            }:

                positions[
                    "sub_district"
                ] = index

            elif header == "village":

                positions[
                    "village"
                ] = index

            elif header in {
                "survey number",
                "survey no.",
                "survey no",
            }:

                positions[
                    "survey_number"
                ] = index

            elif header == "area":

                positions[
                    "area"
                ] = index

            elif header in {
                "description",
                "land description",
            }:

                positions[
                    "description"
                ] = index

        if (
            "serial_number" in positions
            and "district" in positions
            and "village" in positions
            and "survey_number" in positions
            and "area" in positions
        ):

            return positions

    return None


# ============================================================
# SURVEY EXTRACTION
# ============================================================

def extract_surveys(
    soup,
    project_id,
    notification_id,
    notification_metadata,
):

    main_table = find_main_survey_table(
        soup
    )

    if main_table is None:

        raise RuntimeError(
            "Could not find the main survey table."
        )

    header_positions = (
        get_header_positions(
            main_table
        )
    )

    if header_positions is None:

        raise RuntimeError(
            "Could not determine survey table headers."
        )

    rows = main_table.find_all("tr")

    survey_records = []

    party_records = []

    for row_index, row in enumerate(
        rows
    ):

        cells = get_direct_cells(row)

        if len(cells) < 5:
            continue

        cell_values = [
            clean_text(
                cell.get_text(
                    " ",
                    strip=True,
                )
            )
            for cell in cells
        ]

        serial_index = (
            header_positions.get(
                "serial_number"
            )
        )

        if serial_index is None:
            continue

        if serial_index >= len(
            cell_values
        ):
            continue

        serial_number = clean_text(
            cell_values[
                serial_index
            ]
        )

        # Only actual numbered survey rows.
        if not re.fullmatch(
            r"\d+",
            serial_number,
        ):
            continue

        # ----------------------------------------------------
        # Safe getter
        # ----------------------------------------------------

        def get_cell(field_name):

            index = (
                header_positions.get(
                    field_name
                )
            )

            if index is None:
                return ""

            if index >= len(
                cell_values
            ):
                return ""

            return cell_values[index]

        district = get_cell(
            "district"
        )

        sub_district = get_cell(
            "sub_district"
        )

        village = get_cell(
            "village"
        )

        survey_number_raw = get_cell(
            "survey_number"
        )

        area_raw = get_cell(
            "area"
        )

        description_raw = get_cell(
            "description"
        )

        survey_number = (
            clean_survey_number(
                survey_number_raw
            )
        )

        area_hectares = parse_area(
            area_raw
        )

        (
            land_type,
            land_nature,
            land_category,
        ) = parse_land_type(
            description_raw
        )

        # ----------------------------------------------------
        # Land Parties
        # ----------------------------------------------------

        parties = []

        next_row_index = (
            row_index + 1
        )

        if next_row_index < len(
            rows
        ):

            next_row = rows[
                next_row_index
            ]

            next_row_text = clean_text(
                next_row.get_text(
                    " ",
                    strip=True,
                )
            )

            if "Land Parties" in (
                next_row_text
            ):

                parties = (
                    extract_land_parties_from_party_row(
                        next_row
                    )
                )

        # ----------------------------------------------------
        # Party aggregates
        # ----------------------------------------------------

        party_count = len(
            parties
        )

        owner_count = sum(
            1
            for party in parties
            if (
                party.get(
                    "party_type"
                )
                and party[
                    "party_type"
                ].strip().lower()
                == "owner"
            )
        )

        affected_party_count = sum(
            1
            for party in parties
            if (
                party.get(
                    "party_type"
                )
                and party[
                    "party_type"
                ].strip().lower()
                == "affected party"
            )
        )

        total_party_area = 0.0

        valid_party_area_count = 0

        for party in parties:

            party_area = party.get(
                "party_area_hectares"
            )

            if party_area is not None:

                total_party_area += (
                    party_area
                )

                valid_party_area_count += 1

        if (
            valid_party_area_count
            == 0
        ):

            total_party_area = None

        # ----------------------------------------------------
        # Survey record
        # ----------------------------------------------------

        survey_record = {

            "project_id": project_id,

            "notification_id": (
                notification_id
            ),

            "notification_number": (
                notification_metadata.get(
                    "notification_number"
                )
            ),

            "tentative_publish_date": (
                notification_metadata.get(
                    "tentative_publish_date"
                )
            ),

            "serial_number": (
                serial_number
            ),

            "district": (
                district or None
            ),

            "sub_district": (
                sub_district or None
            ),

            "village": (
                village or None
            ),

            "survey_number": (
                survey_number or None
            ),

            "survey_number_raw": (
                survey_number_raw
                or None
            ),

            "area_hectares": (
                area_hectares
            ),

            "area_raw": (
                area_raw or None
            ),

            "land_type": (
                land_type or None
            ),

            "land_nature": (
                land_nature or None
            ),

            "land_category": (
                land_category or None
            ),

            "description_raw": (
                description_raw
                or None
            ),

            "party_count": (
                party_count
            ),

            "owner_count": (
                owner_count
            ),

            "affected_party_count": (
                affected_party_count
            ),

            "total_party_area_hectares": (
                total_party_area
            ),
        }

        survey_records.append(
            survey_record
        )

        # ----------------------------------------------------
        # Detailed party records
        # ----------------------------------------------------

        for party_index, party in enumerate(
            parties,
            start=1,
        ):

            party_records.append(
                {
                    "project_id": (
                        project_id
                    ),

                    "notification_id": (
                        notification_id
                    ),

                    "notification_number": (
                        notification_metadata.get(
                            "notification_number"
                        )
                    ),

                    "survey_serial_number": (
                        serial_number
                    ),

                    "district": (
                        district or None
                    ),

                    "sub_district": (
                        sub_district
                        or None
                    ),

                    "village": (
                        village or None
                    ),

                    "survey_number": (
                        survey_number
                        or None
                    ),

                    "party_sequence": (
                        party_index
                    ),

                    "party_name": (
                        party.get(
                            "party_name"
                        )
                    ),

                    "party_address": (
                        party.get(
                            "party_address"
                        )
                    ),

                    "party_type": (
                        party.get(
                            "party_type"
                        )
                    ),

                    "party_area_hectares": (
                        party.get(
                            "party_area_hectares"
                        )
                    ),

                    "party_area_raw": (
                        party.get(
                            "party_area_raw"
                        )
                    ),
                }
            )

    return (
        survey_records,
        party_records,
    )


# ============================================================
# CSV OUTPUT
# ============================================================

def save_survey_csv(
    records,
    output_file,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "project_id",
        "notification_id",
        "notification_number",
        "tentative_publish_date",
        "serial_number",
        "district",
        "sub_district",
        "village",
        "survey_number",
        "survey_number_raw",
        "area_hectares",
        "area_raw",
        "land_type",
        "land_nature",
        "land_category",
        "description_raw",
        "party_count",
        "owner_count",
        "affected_party_count",
        "total_party_area_hectares",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(records)


def save_party_csv(
    records,
    output_file,
):

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "project_id",
        "notification_id",
        "notification_number",
        "survey_serial_number",
        "district",
        "sub_district",
        "village",
        "survey_number",
        "party_sequence",
        "party_name",
        "party_address",
        "party_type",
        "party_area_hectares",
        "party_area_raw",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(records)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Extract BhoomiRashi 3D survey "
            "and land-party data."
        )
    )

    parser.add_argument(
        "--project-id",
        default=DEFAULT_PROJECT_ID,
        help=(
            "BhoomiRashi project ID "
            "(default: 54635)"
        ),
    )

    parser.add_argument(
        "--notification-id",
        default=DEFAULT_NOTIFICATION_ID,
        help=(
            "BhoomiRashi 3D notification ID "
            "(default: 34193)"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    project_id = str(
        args.project_id
    )

    notification_id = str(
        args.notification_id
    )

    url = (
        BASE_URL
        + "sdet1.cshtml"
        + f"?project_id={project_id}"
        + "&EncHid="
        + f"&notification_id={notification_id}"
        + "&nid=9"
    )

    print("=" * 70)
    print(
        "BHOOMIRASHI 3D SURVEY EXTRACTOR"
    )
    print("=" * 70)

    print(
        f"Project ID       : {project_id}"
    )

    print(
        f"Notification ID  : {notification_id}"
    )

    print(
        f"URL              : {url}"
    )

    print()

    # --------------------------------------------------------
    # Fetch
    # --------------------------------------------------------

    print("Fetching page...")

    try:

        html = fetch_page(url)

    except Exception as exc:

        print(
            f"ERROR: Failed to fetch page: {exc}"
        )

        return

    print(
        "HTTP page fetched successfully."
    )

    print(
        f"Content length: {len(html):,} bytes"
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    print()
    print(
        "Extracting notification metadata..."
    )

    metadata = (
        extract_notification_metadata(
            soup
        )
    )

    print(
        "Notification Number : "
        f"{metadata['notification_number']}"
    )

    print(
        "Tentative Publish Date : "
        f"{metadata['tentative_publish_date']}"
    )

    # --------------------------------------------------------
    # Surveys
    # --------------------------------------------------------

    print()
    print(
        "Searching for main survey table..."
    )

    try:

        (
            survey_records,
            party_records,
        ) = extract_surveys(
            soup,
            project_id,
            notification_id,
            metadata,
        )

    except Exception as exc:

        print(
            f"ERROR: {exc}"
        )

        return

    print(
        "Survey records extracted: "
        f"{len(survey_records)}"
    )

    # --------------------------------------------------------
    # First 5
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("FIRST 5 SURVEY RECORDS")
    print("-" * 70)

    for record in survey_records[:5]:

        print()

        print(
            f"Survey #{record['serial_number']}"
        )

        print(
            f"  District       : "
            f"{record['district']}"
        )

        print(
            f"  Sub District   : "
            f"{record['sub_district']}"
        )

        print(
            f"  Village        : "
            f"{record['village']}"
        )

        print(
            f"  Survey Number  : "
            f"{record['survey_number']}"
        )

        print(
            f"  Area (ha)      : "
            f"{record['area_hectares']}"
        )

        print(
            f"  Land Type      : "
            f"{record['land_type']}"
        )

        print(
            f"  Land Nature    : "
            f"{record['land_nature']}"
        )

        print(
            f"  Land Category  : "
            f"{record['land_category']}"
        )

        print(
            f"  Party Count    : "
            f"{record['party_count']}"
        )

        print(
            f"  Owner Count    : "
            f"{record['owner_count']}"
        )

        print(
            f"  Affected Party : "
            f"{record['affected_party_count']}"
        )

        print(
            f"  Party Area (ha): "
            f"{record['total_party_area_hectares']}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_with_parties = sum(
        1
        for record in survey_records
        if record["party_count"] > 0
    )

    total_area = sum(
        record["area_hectares"]
        for record in survey_records
        if record["area_hectares"]
        is not None
    )

    total_owners = sum(
        record["owner_count"]
        for record in survey_records
    )

    total_affected = sum(
        record[
            "affected_party_count"
        ]
        for record in survey_records
    )

    print()
    print("-" * 70)
    print("EXTRACTION SUMMARY")
    print("-" * 70)

    print(
        f"Survey records       : "
        f"{len(survey_records)}"
    )

    print(
        f"Surveys with parties : "
        f"{total_with_parties}"
    )

    print(
        f"Total party records  : "
        f"{len(party_records)}"
    )

    print(
        f"Owner records        : "
        f"{total_owners}"
    )

    print(
        f"Affected parties     : "
        f"{total_affected}"
    )

    print(
        f"Total survey area ha : "
        f"{total_area:.6f}"
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    project_output_dir = (
        OUTPUT_DIR
        / project_id
    )

    survey_output = (
        project_output_dir
        / f"3d_{notification_id}_surveys.csv"
    )

    party_output = (
        project_output_dir
        / f"3d_{notification_id}_land_parties.csv"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print()
    print(
        "Saving normalized files..."
    )

    save_survey_csv(
        survey_records,
        survey_output,
    )

    save_party_csv(
        party_records,
        party_output,
    )

    print()
    print(
        "Survey CSV:"
    )

    print(
        survey_output
    )

    print()
    print(
        "Land Parties CSV:"
    )

    print(
        party_output
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()