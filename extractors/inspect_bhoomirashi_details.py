import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin


BASE_URL = "https://bhoomirashi.gov.in/auth/revamp/"

URLS = {
    "3a_villages": (
        "calavil.cshtml?"
        "project_id=54635&"
        "EncHid=&"
        "notification_id=50840&"
        "nid=9"
    ),

    "3A_surveys": (
        "sdet.cshtml?"
        "project_id=54635&"
        "EncHid=&"
        "notification_id=44531&"
        "nid=9"
    ),

    "3A_objections": (
        "three_a_obj_rep.cshtml?"
        "notification_id=44531&"
        "notification_type_id=1"
    ),

    "3D_surveys": (
        "sdet1.cshtml?"
        "project_id=54635&"
        "EncHid=&"
        "notification_id=34193&"
        "nid=9"
    ),
}


def inspect_page(name, relative_url):

    url = urljoin(BASE_URL, relative_url)

    print("\n")
    print("=" * 80)
    print(name)
    print("=" * 80)

    print("URL:")
    print(url)

    response = requests.get(
        url,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    print("\nHTTP Status:", response.status_code)
    print("Content-Type:", response.headers.get("Content-Type"))
    print("Content-Length:", len(response.content))

    if response.status_code != 200:
        print("FAILED")
        return

    soup = BeautifulSoup(
        response.text,
        "lxml"
    )

    print("\nTITLE:")
    print(
        soup.title.get_text(
            " ",
            strip=True
        )
        if soup.title
        else "No title"
    )

    tables = soup.find_all("table")

    print("\nNUMBER OF TABLES:", len(tables))

    for table_number, table in enumerate(
        tables,
        start=1
    ):

        print("\n" + "-" * 80)
        print(f"TABLE {table_number}")
        print("-" * 80)

        rows = table.find_all("tr")

        for row in rows[:15]:

            cells = row.find_all(
                ["th", "td"]
            )

            values = [
                cell.get_text(
                    " ",
                    strip=True
                )
                for cell in cells
            ]

            if values:
                print(values)


for name, relative_url in URLS.items():

    inspect_page(
        name,
        relative_url
    )