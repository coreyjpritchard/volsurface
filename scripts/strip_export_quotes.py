"""Remove per-quote market points from a static marimo export of a session.

Run from the repository root after exporting, for example:

    marimo export html sessions/00_the_smile.py -o docs/00_the_smile.html
    python scripts/strip_export_quotes.py docs/00_the_smile.html

The Cboe quotes are not redistributed, so a committed export must not carry the
market implied vol of each quote. This drops every plotly trace named "SPX" except
in the at-the-money skew figure, whose SPX points are one level and skew per expiry,
and unticks the "Overlay SPX market implied vols" box so the page matches what it
shows. Everything else in the file is kept byte for byte.
"""

from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

FIGURE = re.compile(
    r"(marimo-plotly data-initial-value='\{\}' data-label='null' data-figure=')([^']*)'"
)
CHECKBOX = "marimo-checkbox data-initial-value='true'"
MARKET = "SPX"
SUMMARY_TITLE = "At-the-money skew"


def _decode(raw: str) -> dict:
    text = re.sub(r"\\u([0-9A-Fa-f]{4})", lambda m: chr(int(m.group(1), 16)), raw)
    return json.loads(html.unescape(text))


def _encode(figure: dict) -> str:
    text = html.escape(json.dumps(figure, separators=(",", ":")), quote=True)
    return text.replace("&", "\\u0026").replace("<", "\\u003C").replace(">", "\\u003E")


def figures(page: str) -> list[dict]:
    """The plotly figures embedded in an export."""
    return [_decode(m.group(2)) for m in FIGURE.finditer(page)]


def title(figure: dict) -> str:
    value = figure.get("layout", {}).get("title")
    return (value.get("text") if isinstance(value, dict) else value) or ""


def is_per_quote(figure: dict, trace: dict) -> bool:
    return trace.get("name") == MARKET and not title(figure).startswith(SUMMARY_TITLE)


def strip(page: str) -> str:
    def replace(match: re.Match) -> str:
        figure = _decode(match.group(2))
        figure["data"] = [t for t in figure["data"] if not is_per_quote(figure, t)]
        return match.group(1) + _encode(figure) + "'"

    page = FIGURE.sub(replace, page)
    return page.replace(CHECKBOX, "marimo-checkbox data-initial-value='false'")


def main(paths: list[str]) -> None:
    for name in paths:
        path = Path(name)
        page = path.read_text(encoding="utf-8")
        path.write_text(strip(page), encoding="utf-8")
        kept = sum(len(f["data"]) for f in figures(strip(page)))
        print(f"{path}: {len(figures(page))} figures, {kept} traces kept")


if __name__ == "__main__":
    main(sys.argv[1:])
