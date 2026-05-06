"""Thin AniList GraphQL client with rate-limit handling and shape validation."""
from __future__ import annotations

import time

import requests

ANILIST_URL = "https://graphql.anilist.co"
MAX_RETRIES = 5
INITIAL_BACKOFF_SEC = 2.0


class AniListError(RuntimeError):
    pass


def post(query: str, variables: dict) -> dict:
    """POST a GraphQL query, retrying on 429 with exponential backoff. Returns the `data` field."""
    backoff = INITIAL_BACKOFF_SEC
    for attempt in range(1, MAX_RETRIES + 1):
        resp = requests.post(ANILIST_URL, json={"query": query, "variables": variables}, timeout=30)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", backoff))
            print(f"  rate-limited (attempt {attempt}/{MAX_RETRIES}) — sleeping {retry_after:.1f}s")
            time.sleep(retry_after)
            backoff *= 2
            continue
        if resp.status_code != 200:
            raise AniListError(f"AniList HTTP {resp.status_code}: {resp.text[:500]}")
        body = resp.json()
        if "errors" in body:
            raise AniListError(f"AniList GraphQL errors: {body['errors']}")
        if "data" not in body:
            raise AniListError(f"AniList response missing 'data': {body}")
        return body["data"]
    raise AniListError(f"AniList rate-limited after {MAX_RETRIES} retries")


def collect_entries(data: dict) -> list[dict]:
    """Flatten MediaListCollection.lists[].entries[] into a single entry list. Validates shape."""
    if not isinstance(data, dict) or "MediaListCollection" not in data:
        raise AniListError(f"unexpected response shape: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    collection = data["MediaListCollection"] or {}
    lists = collection.get("lists") or []
    entries: list[dict] = []
    for lst in lists:
        for entry in lst.get("entries", []) or []:
            entries.append(entry)
    return entries
