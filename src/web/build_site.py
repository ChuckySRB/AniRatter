"""Step 5 — Stage predictions for the web page.

- Copies data/predictions/predictions.json → web/predictions.json so the page can fetch() it
  when served over HTTP.
- Also injects the JSON inline into web/index.html (replacing the placeholder
  `<script id="predictions-data" …>{}</script>` block) so the page works opened directly via
  file:// (browsers block fetch() on file://).

Run: python -m src.web.build_site
"""
from __future__ import annotations

import json
import re
import shutil

import config

INLINE_RE = re.compile(
    r'(<script\s+id="predictions-data"[^>]*>)(.*?)(</script>)',
    re.DOTALL,
)


def main() -> None:
    if not config.PREDICTIONS_JSON.exists():
        raise SystemExit(f"missing {config.PREDICTIONS_JSON} — run `python -m src.predict.predict` first")
    config.WEB_DIR.mkdir(parents=True, exist_ok=True)
    target_json = config.WEB_DIR / "predictions.json"
    shutil.copy2(config.PREDICTIONS_JSON, target_json)
    print(f"copied {config.PREDICTIONS_JSON.name} → {target_json}")

    index = config.WEB_DIR / "index.html"
    if index.exists():
        html = index.read_text(encoding="utf-8")
        payload = config.PREDICTIONS_JSON.read_text(encoding="utf-8")
        if INLINE_RE.search(html):
            new_html = INLINE_RE.sub(
                lambda m: m.group(1) + payload + m.group(3),
                html,
                count=1,
            )
            index.write_text(new_html, encoding="utf-8")
            print(f"inlined predictions into {index}")
        else:
            print("  note: no <script id=\"predictions-data\"> block in index.html; skipping inline.")

    print("\nopen web/index.html in your browser, or run:  python -m src.web.serve")


if __name__ == "__main__":
    main()
