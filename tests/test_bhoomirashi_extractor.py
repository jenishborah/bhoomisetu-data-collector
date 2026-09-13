from extractors.bhoomirashi_html import BhoomiRashiExtractor
import json


HTML_FILE = "output/raw/bhoomirashi/54635/source.html"


extractor = BhoomiRashiExtractor(HTML_FILE)

data = extractor.extract_all()

print(
    json.dumps(
        data,
        indent=2,
        ensure_ascii=False
    )
)