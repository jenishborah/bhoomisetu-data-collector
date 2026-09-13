from pathlib import Path
from bs4 import BeautifulSoup


HTML_FILE = Path(
    "output/raw/bhoomirashi/54635/source.html"
)

html = HTML_FILE.read_text(
    encoding="utf-8"
)

soup = BeautifulSoup(
    html,
    "lxml"
)


for table_number, table in enumerate(
    soup.find_all("table"),
    start=1
):

    print("\n" + "=" * 80)
    print(f"TABLE {table_number}")
    print("=" * 80)

    for row_number, row in enumerate(
        table.find_all("tr"),
        start=1
    ):

        cells = row.find_all(["th", "td"])

        for cell_number, cell in enumerate(
            cells,
            start=1
        ):

            links = cell.find_all("a")

            for link in links:

                text = link.get_text(
                    " ",
                    strip=True
                )

                href = link.get("href")

                print(
                    f"Row {row_number}, "
                    f"Cell {cell_number}: "
                    f"{text!r}"
                )

                print(
                    f"    href = {href!r}"
                )