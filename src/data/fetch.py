"""Step 1 — Fetch raw AniList data for the configured user.

Writes 4 JSON files to data/raw/:
  rated_anime.json, rated_manga.json, planning_anime.json, planning_manga.json

Each file is a flat list of entry dicts (the API's nested lists[].entries[] wrapper is stripped).
Entries are deduplicated by media.id.

Run: python -m src.data.fetch
"""
from __future__ import annotations

import json
from pathlib import Path

import config
from src.anilist.client import collect_entries, post
from src.anilist.queries import MEDIA_LIST_QUERY, variables_planning, variables_rated


def _dedupe_by_media_id(entries: list[dict]) -> tuple[list[dict], int]:
    seen: set[int] = set()
    out: list[dict] = []
    for entry in entries:
        media = entry.get("media") or {}
        mid = media.get("id")
        if mid is None or mid in seen:
            continue
        seen.add(mid)
        out.append(entry)
    return out, len(entries) - len(out)


def _fetch_segment(label: str, variables: dict, out_path: Path) -> None:
    print(f"fetching {label} …")
    data = post(MEDIA_LIST_QUERY, variables)
    raw = collect_entries(data)
    deduped, dropped = _dedupe_by_media_id(raw)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(deduped, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  → {len(deduped)} entries (deduped {dropped}) → {out_path}")


def main() -> None:
    user = config.USER
    print(f"AniList user: {user}")
    _fetch_segment("rated anime", variables_rated(user, "ANIME"), config.RAW_RATED_ANIME)
    _fetch_segment("rated manga", variables_rated(user, "MANGA"), config.RAW_RATED_MANGA)
    _fetch_segment("planning anime", variables_planning(user, "ANIME"), config.RAW_PLANNING_ANIME)
    _fetch_segment("planning manga", variables_planning(user, "MANGA"), config.RAW_PLANNING_MANGA)
    print("done.")


if __name__ == "__main__":
    main()
