"""Step 2 — Build training features from rated raw data.

Reads:  data/raw/rated_anime.json, rated_manga.json
Writes: data/processed/features_train.parquet, feature_spec.json, feature_spec.joblib

Run: python -m src.preprocess.build_features
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config
from src.preprocess.feature_builder import (
    FeatureBuilder,
    entries_to_dataframe,
    save_spec,
)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run `python -m src.data.fetch` first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rated_anime = _load(config.RAW_RATED_ANIME)
    rated_manga = _load(config.RAW_RATED_MANGA)
    entries = rated_anime + rated_manga
    print(f"loaded {len(rated_anime)} rated anime + {len(rated_manga)} rated manga = {len(entries)} entries")

    df = entries_to_dataframe(entries, min_tag_rank=config.MIN_TAG_RANK)
    print(f"flattened to {len(df)} rows")

    # Filter: must have a user score in (0, 100] and a meanScore for the item.
    before = len(df)
    df = df[(df["score"] > 0) & (df["score"] <= 100) & df["meanScore"].notna()].reset_index(drop=True)
    print(f"filtered to {len(df)} rows (dropped {before - len(df)} with score==0 or missing meanScore)")
    if len(df) == 0:
        raise SystemExit("no rated items left after filtering — check the raw data")

    builder = FeatureBuilder()
    X, spec = builder.fit_transform(df)
    print(f"feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"  numerical: {len(spec.numerical_cols)} | format: {len(spec.format_categories)} | "
          f"source: {len(spec.source_categories)} | genres: {len(spec.genre_vocab)} | tags: {len(spec.tag_vocab)}")

    out = X.copy()
    out["score"] = df["score"].astype(float).values

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(config.FEATURES_PARQUET, index=False)
    save_spec(spec, config.FEATURE_SPEC_JSON, config.FEATURE_SPEC_JOBLIB)
    print(f"wrote {config.FEATURES_PARQUET}")
    print(f"wrote {config.FEATURE_SPEC_JSON}")
    print(f"wrote {config.FEATURE_SPEC_JOBLIB}")
    print(f"spec_hash: {spec.spec_hash[:12]}…")


if __name__ == "__main__":
    main()
