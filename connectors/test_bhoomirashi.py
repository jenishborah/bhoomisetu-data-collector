import requests
from pathlib import Path
from datetime import datetime, timezone

URL = "https://bhoomirashi.gov.in/auth/revamp/prep_public.cshtml?nid=1&project_id=54635"

OUTPUT_DIR = Path("output/raw/bhoomirashi/54635")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

response = requests.get(
    URL,
    timeout=30,
    headers={
        "User-Agent": "Mozilla/5.0"
    }
)

print("Status:", response.status_code)

if response.status_code != 200:
    raise RuntimeError(
        f"Failed to retrieve page: HTTP {response.status_code}"
    )

# Save original source exactly as received
source_file = OUTPUT_DIR / "source.html"
source_file.write_text(
    response.text,
    encoding="utf-8"
)

# Save retrieval metadata
metadata = f"""source_portal=BhoomiRashi
source_url={URL}
source_record_id=54635
retrieved_at={datetime.now(timezone.utc).isoformat()}
http_status={response.status_code}
"""

metadata_file = OUTPUT_DIR / "metadata.txt"
metadata_file.write_text(
    metadata,
    encoding="utf-8"
)

print("Saved:")
print(source_file)
print(metadata_file)