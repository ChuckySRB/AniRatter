"""FeatureBuilder — fit/transform a tabular feature matrix from raw AniList entries.

The same .transform() code path is used at training time and prediction time. Fit captures all
parameters (vocabularies, scaler stats, column order) into a FeatureSpec; predict replays them.

Numerical features  : meanScore, averageScore, popularity, favourites, seasonYear, mean_top_tag_rank
                       — z-scored with persisted mean/std; missing → train median.
Single binary       : is_manga (from media.type).
One-hot             : format, source — fixed vocab + always-present "_other" bucket.
Multi-label binary  : genres, tags(rank>=MIN_TAG_RANK).

Returns features sorted by FeatureSpec.column_order. The 'score' target column is appended at fit
time (because training data has it) but is NOT in column_order.
"""
from __future__ import annotations

import hashlib
import json
import sklearn
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

import config

NUMERICAL_COLS = [
    "meanScore",
    "averageScore",
    "popularity",
    "favourites",
    "seasonYear",
    "mean_top_tag_rank",
]
OTHER_BUCKET = "_other"


@dataclass
class FeatureSpec:
    column_order: list[str]
    numerical_cols: list[str]
    numerical_mean: dict[str, float]
    numerical_std: dict[str, float]
    numerical_median: dict[str, float]
    format_categories: list[str]
    source_categories: list[str]
    genre_vocab: list[str]
    tag_vocab: list[str]
    min_tag_rank: int
    sklearn_version: str
    spec_hash: str = ""

    def to_json_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_json_dict(cls, d: dict) -> "FeatureSpec":
        return cls(**d)


def _compute_hash(d: dict) -> str:
    payload = {k: v for k, v in d.items() if k != "spec_hash"}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Raw → flat dataframe
# ---------------------------------------------------------------------------

def entries_to_dataframe(entries: list[dict], min_tag_rank: int) -> pd.DataFrame:
    """Flatten raw API entries into a per-row dict, including the 'score' (0 if unrated)."""
    rows: list[dict] = []
    for entry in entries:
        media = entry.get("media") or {}
        tags_full = media.get("tags") or []
        kept_tags = [t for t in tags_full if (t.get("rank") or 0) >= min_tag_rank]
        kept_names = [t["name"] for t in kept_tags if t.get("name")]
        ranks = [t.get("rank") or 0 for t in kept_tags]
        mean_top_rank = float(np.mean(ranks)) if ranks else 0.0
        start = media.get("startDate") or {}
        title = media.get("title") or {}
        cover = media.get("coverImage") or {}
        rows.append({
            "id": media.get("id"),
            "title_romaji": title.get("romaji"),
            "title_english": title.get("english"),
            "title_native": title.get("native"),
            "cover_image": cover.get("large"),
            "site_url": media.get("siteUrl"),
            "type": media.get("type"),
            "format": media.get("format"),
            "source": media.get("source"),
            "seasonYear": start.get("year"),
            "genres": list(media.get("genres") or []),
            "tags": kept_names,
            "mean_top_tag_rank": mean_top_rank,
            "meanScore": media.get("meanScore"),
            "averageScore": media.get("averageScore"),
            "popularity": media.get("popularity"),
            "favourites": media.get("favourites"),
            "episodes": media.get("episodes"),
            "chapters": media.get("chapters"),
            "score": entry.get("score") or 0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FeatureBuilder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """Fits + applies the feature transform. Stateful only via FeatureSpec."""

    # ---- fit ---- #
    def fit(self, df: pd.DataFrame) -> FeatureSpec:
        # Numerical stats
        numerical_mean: dict[str, float] = {}
        numerical_std: dict[str, float] = {}
        numerical_median: dict[str, float] = {}
        for col in NUMERICAL_COLS:
            series = pd.to_numeric(df[col], errors="coerce")
            median = float(series.median()) if series.notna().any() else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            if std == 0.0 or np.isnan(std):
                std = 1.0
            numerical_median[col] = median
            numerical_mean[col] = mean
            numerical_std[col] = std

        # Categorical vocabularies — always include the "_other" bucket so unseen predict-time
        # values map somewhere real instead of silently producing all-zero one-hots.
        format_cats = sorted({v for v in df["format"].dropna().unique()}) + [OTHER_BUCKET]
        source_cats = sorted({v for v in df["source"].dropna().unique()}) + [OTHER_BUCKET]

        genre_vocab = sorted({g for gs in df["genres"] for g in (gs or [])})
        tag_vocab = sorted({t for ts in df["tags"] for t in (ts or [])})

        # Column order: numerical → is_manga → format one-hots → source one-hots → genres → tags
        column_order: list[str] = []
        column_order.extend(NUMERICAL_COLS)
        column_order.append("is_manga")
        column_order.extend([f"format_{c}" for c in format_cats])
        column_order.extend([f"source_{c}" for c in source_cats])
        column_order.extend([f"genre_{g}" for g in genre_vocab])
        column_order.extend([f"tag_{t}" for t in tag_vocab])

        spec = FeatureSpec(
            column_order=column_order,
            numerical_cols=NUMERICAL_COLS,
            numerical_mean=numerical_mean,
            numerical_std=numerical_std,
            numerical_median=numerical_median,
            format_categories=format_cats,
            source_categories=source_cats,
            genre_vocab=genre_vocab,
            tag_vocab=tag_vocab,
            min_tag_rank=config.MIN_TAG_RANK,
            sklearn_version=sklearn.__version__,
        )
        spec.spec_hash = _compute_hash(spec.to_json_dict())
        return spec

    # ---- transform ---- #
    def transform(self, df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
        n = len(df)
        cols: dict[str, np.ndarray] = {}

        # Numerical → fillna(median), z-score
        for col in spec.numerical_cols:
            series = pd.to_numeric(df[col], errors="coerce").fillna(spec.numerical_median[col])
            cols[col] = ((series - spec.numerical_mean[col]) / spec.numerical_std[col]).to_numpy()

        # is_manga
        cols["is_manga"] = (df["type"].astype(str) == "MANGA").astype(np.int8).to_numpy()

        # format one-hot — values not in vocab → "_other"
        format_known = set(spec.format_categories) - {OTHER_BUCKET}
        format_resolved = df["format"].where(df["format"].isin(format_known), OTHER_BUCKET).to_numpy()
        for c in spec.format_categories:
            cols[f"format_{c}"] = (format_resolved == c).astype(np.int8)

        # source one-hot
        source_known = set(spec.source_categories) - {OTHER_BUCKET}
        source_resolved = df["source"].where(df["source"].isin(source_known), OTHER_BUCKET).to_numpy()
        for c in spec.source_categories:
            cols[f"source_{c}"] = (source_resolved == c).astype(np.int8)

        # genres — vectorised: one row-set per item, then membership tests per vocab term
        genre_sets = [set(g or []) for g in df["genres"]]
        genre_vocab_set = set(spec.genre_vocab)
        for g in spec.genre_vocab:
            cols[f"genre_{g}"] = np.fromiter((g in s for s in genre_sets), dtype=np.int8, count=n)
        unknown_genre_rows = sum(1 for s in genre_sets if s and not s.issubset(genre_vocab_set))
        if unknown_genre_rows:
            print(f"  [transform] {unknown_genre_rows}/{n} items had genres outside training vocab (silently dropped)")

        # tags — same pattern
        tag_sets = [set(t or []) for t in df["tags"]]
        tag_vocab_set = set(spec.tag_vocab)
        for t in spec.tag_vocab:
            cols[f"tag_{t}"] = np.fromiter((t in s for s in tag_sets), dtype=np.int8, count=n)
        unknown_tag_rows = sum(1 for s in tag_sets if s and not s.issubset(tag_vocab_set))
        if unknown_tag_rows:
            print(f"  [transform] {unknown_tag_rows}/{n} items had tags outside training vocab (silently dropped)")

        # Build the dataframe in one shot in spec column order — guarantees train/predict alignment
        return pd.DataFrame({c: cols[c] for c in spec.column_order}, index=df.index)

    # ---- fit_transform ---- #
    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, FeatureSpec]:
        spec = self.fit(df)
        return self.transform(df, spec), spec


# ---------------------------------------------------------------------------
# Spec persistence
# ---------------------------------------------------------------------------

def save_spec(spec: FeatureSpec, json_path, joblib_path) -> None:
    import joblib
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(spec.to_json_dict(), indent=2), encoding="utf-8")
    joblib.dump(spec, joblib_path)


def load_spec(json_path=None, joblib_path=None) -> FeatureSpec:
    """Prefer joblib (round-trip safe). Fall back to JSON."""
    import joblib
    if joblib_path is not None and joblib_path.exists():
        return joblib.load(joblib_path)
    if json_path is not None and json_path.exists():
        return FeatureSpec.from_json_dict(json.loads(json_path.read_text(encoding="utf-8")))
    raise FileNotFoundError("no FeatureSpec found")


__all__ = [
    "FeatureBuilder",
    "FeatureSpec",
    "entries_to_dataframe",
    "save_spec",
    "load_spec",
    "NUMERICAL_COLS",
    "OTHER_BUCKET",
]
