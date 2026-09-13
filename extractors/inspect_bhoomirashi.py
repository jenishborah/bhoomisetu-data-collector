from pathlib import Path
from bs4 import BeautifulSoup

HTML_FILE = Path("output/raw/bhoomirashi/54635/source.html")

html = HTML_FILE.read_text(encoding="utf-8")

soup = BeautifulSoup(html, "lxml")

print("Title:")
print(soup.title.get_text(" ", strip=True) if soup.title else "No title")

print("\nNumber of tables:", len(soup.find_all("table")))

for i, table in enumerate(soup.find_all("table"), start=1):
    rows = table.find_all("tr")

    print(f"\n{'=' * 60}")
    print(f"TABLE {i}")
    print(f"Rows: {len(rows)}")
    print(f"{'=' * 60}")

    for row in rows[:10]:
        cells = row.find_all(["th", "td"])

        values = [
            cell.get_text(" ", strip=True)
            for cell in cells
        ]

        print(values)