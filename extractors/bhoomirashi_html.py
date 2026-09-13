from pathlib import Path
from bs4 import BeautifulSoup
import re


class BhoomiRashiExtractor:

    def __init__(self, html_path):
        self.html_path = Path(html_path)

        if not self.html_path.exists():
            raise FileNotFoundError(
                f"HTML file not found: {self.html_path}"
            )

        self.html = self.html_path.read_text(
            encoding="utf-8"
        )

        self.soup = BeautifulSoup(
            self.html,
            "lxml"
        )

        self.tables = self.soup.find_all("table")

    def clean_text(self, value):
        """
        Clean whitespace without changing the actual meaning
        of the source data.
        """

        if value is None:
            return None

        value = value.replace("\xa0", " ")
        value = re.sub(r"\s+", " ", value)

        return value.strip()

    def get_table_rows(self, table):
        """
        Convert an HTML table into a list of rows.
        """

        rows = []

        for tr in table.find_all("tr"):

            cells = tr.find_all(["th", "td"])

            values = [
                self.clean_text(cell.get_text(" ", strip=True))
                for cell in cells
            ]

            if values:
                rows.append(values)

        return rows

    def extract_basic_details(self):

        if len(self.tables) < 1:
            return {}

        rows = self.get_table_rows(self.tables[0])

        data = {}

        for row in rows:

            if len(row) >= 2:

                key = row[0].lower().strip()
                value = row[1]

                if key == "project name":
                    data["project_name"] = value

                elif key == "project number":
                    data["project_number"] = value

            # Land Required / Land Available
            if len(row) >= 4:

                if row[0].lower() == "land required":

                    data["land_required_raw"] = row[1]
                    data["land_available_raw"] = row[3]

            # Land to be acquired / Land acquired
            if len(row) >= 4:

                if row[0].lower() == "land to be acquired":

                    data["land_to_acquire_raw"] = row[1]
                    data["land_acquired_raw"] = row[3]

        return data

    def extract_sanction_details(self):

        if len(self.tables) < 2:
            return []

        rows = self.get_table_rows(self.tables[1])

        records = []

        # Skip:
        # row 0 = title
        # row 1 = column headers

        for row in rows[2:]:

            if not row:
                continue

            # Ignore total row
            if row[0].lower() == "total":
                continue

            record = {
                "sanction_number": row[0] if len(row) > 0 else None,
                "sanction_date": row[1] if len(row) > 1 else None,
                "sanction_amount_raw": row[2] if len(row) > 2 else None,
            }

            records.append(record)

        return records

    def extract_3a_notifications(self):

        if len(self.tables) < 3:
            return []

        rows = self.get_table_rows(self.tables[2])

        records = []

        # Skip title and header
        for row in rows[2:]:

            if len(row) < 5:
                continue

            record = {
                "sno": row[0],
                "publish_date": row[1],
                "notification_number": row[2],
                "cala_villages": row[3],
                "status": row[4],
            }

            records.append(record)

        return records

    def extract_3A_notifications(self):

        if len(self.tables) < 4:
            return []

        rows = self.get_table_rows(self.tables[3])

        records = []

        for row in rows[2:]:

            if len(row) < 6:
                continue

            record = {
                "sno": row[0],
                "publish_date": row[1],
                "notification_number": row[2],
                "survey_numbers": row[3],
                "objections": row[4],
                "status": row[5],
            }

            records.append(record)

        return records

    def extract_3D_notifications(self):

        if len(self.tables) < 5:
            return []

        rows = self.get_table_rows(self.tables[4])

        records = []

        for row in rows[2:]:

            if len(row) < 5:
                continue

            record = {
                "sno": row[0],
                "publish_date": row[1],
                "notification_number": row[2],
                "survey_numbers": row[3],
                "status": row[4],
            }

            records.append(record)

        return records

    def extract_consent_purchase(self):

        if len(self.tables) < 6:
            return []

        rows = self.get_table_rows(self.tables[5])

        records = []

        # Skip title and header
        for row in rows[2:]:

            if not row:
                continue

            record = {
                "state": row[0] if len(row) > 0 else None,
                "district": row[1] if len(row) > 1 else None,
                "sub_district": row[2] if len(row) > 2 else None,
                "villages": row[3] if len(row) > 3 else None,
                "survey_number": row[4] if len(row) > 4 else None,
                "land_party": row[5] if len(row) > 5 else None,
                "amount": row[6] if len(row) > 6 else None,
                "disbursement_details": row[7] if len(row) > 7 else None,
            }

            records.append(record)

        return records

    def extract_all(self):

        return {
            "basic_details": self.extract_basic_details(),
            "sanction_details": self.extract_sanction_details(),
            "3a_notifications": self.extract_3a_notifications(),
            "3A_notifications": self.extract_3A_notifications(),
            "3D_notifications": self.extract_3D_notifications(),
            "consent_purchase": self.extract_consent_purchase(),
        }