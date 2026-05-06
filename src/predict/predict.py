"""Step 4 — Run the saved model on planning items, compute rec_score, write predictions.json.

Reads:  data/raw/planning_anime.json, planning_manga.json
        data/processed/feature_spec.json/.joblib
        data/models/best_model.joblib, metrics.json
Writes: data/predictions/predictions.json + dated copy

Run: python -m src.predict.predict
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn

import config
from src.preprocess.feature_builder import (
    FeatureBuilder,
    entries_to_dataframe,
    load_spec,
)


def _load_json(path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"missing {path} — run `python -m src.data.fetch` first")
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_rec(predicted: np.ndarray, mean_score: np.ndarray) -> tuple[np.ndarray, float]:
    raw = (predicted - config.REC_BASE_OFFSET) + config.REC_PERSONAL_WEIGHT * (predicted - mean_score)
    clipped = np.clip(raw, -config.REC_CLIP, config.REC_CLIP)
    clip_fraction = float(np.mean(np.abs(raw) > config.REC_CLIP)) if len(raw) else 0.0
    return clipped, clip_fraction


def main() -> None:
    if not config.MODEL_JOBLIB.exists():
        raise SystemExit(f"missing {config.MODEL_JOBLIB} — run `python -m src.train.train` first")

    spec = load_spec(config.FEATURE_SPEC_JSON, config.FEATURE_SPEC_JOBLIB)
    metrics = json.loads(config.METRICS_JSON.read_text(encoding="utf-8")) if config.METRICS_JSON.exists() else {}

    if metrics.get("spec_hash") and metrics["spec_hash"] != spec.spec_hash:
        raise SystemExit(
            f"spec_hash mismatch: model trained with {metrics['spec_hash'][:12]}…, "
            f"current spec is {spec.spec_hash[:12]}…  — re-run training."
        )
    if metrics.get("sklearn_version") and metrics["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"sklearn version mismatch: model trained on {metrics['sklearn_version']}, "
            f"current is {sklearn.__version__} — pickle may misbehave."
        )

    model = joblib.load(config.MODEL_JOBLIB)

    planning = _load_json(config.RAW_PLANNING_ANIME) + _load_json(config.RAW_PLANNING_MANGA)
    print(f"loaded {len(planning)} planning entries")
    df = entries_to_dataframe(planning, min_tag_rank=spec.min_tag_rank)

    before = len(df)
    df = df[df["meanScore"].notna()].reset_index(drop=True)
    print(f"filtered to {len(df)} (dropped {before - len(df)} with no meanScore)")
    if len(df) == 0:
        raise SystemExit("no planning items left to predict on")

    X = FeatureBuilder().transform(df, spec)
    P = model.predict(X[spec.column_order].values)
    mean_score = df["meanScore"].astype(float).values
    rec, clip_fraction = _compute_rec(P, mean_score)

    print(f"predicted scores: min {P.min():.2f} max {P.max():.2f} mean {P.mean():.2f}")
    print(f"rec_score clip fraction: {clip_fraction:.1%}")
    if clip_fraction > 0.20:
        print("  WARNING: >20% of items hit the clip — formula weights may need revisiting.")

    items: list[dict] = []
    for i, row in df.iterrows():
        items.append({
            "id": int(row["id"]) if pd.notna(row["id"]) else None,
            "title_romaji": row["title_romaji"],
            "title_english": row["title_english"],
            "title_native": row["title_native"],
            "cover_image": row["cover_image"],
            "site_url": row["site_url"],
            "type": row["type"],
            "format": row["format"],
            "year": int(row["seasonYear"]) if pd.notna(row["seasonYear"]) else None,
            "source": row["source"],
            "genres": list(row["genres"] or []),
            "episodes": int(row["episodes"]) if pd.notna(row["episodes"]) else None,
            "chapters": int(row["chapters"]) if pd.notna(row["chapters"]) else None,
            "predicted_score": round(float(P[i]), 2),
            "mean_score": int(row["meanScore"]),
            "rec_score": round(float(rec[i]), 2),
        })

    items.sort(key=lambda x: x["rec_score"], reverse=True)

    payload = {
        "schema_version": 1,
        "metadata": {
            "user": config.USER,
            "model_name": metrics.get("model_name"),
            "test_mae": metrics.get("test_mae"),
            "test_r2": metrics.get("test_r2"),
            "trained_at": metrics.get("trained_at"),
            "predicted_at": datetime.now(timezone.utc).isoformat(),
            "n_items": len(items),
            "rec_clip_fraction": clip_fraction,
            "rec_formula": f"clip( (P - {config.REC_BASE_OFFSET}) + {config.REC_PERSONAL_WEIGHT} * (P - meanScore), -{config.REC_CLIP}, {config.REC_CLIP} )",
            "git_commit": metrics.get("git_commit"),
        },
        "items": items,
    }

    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    config.PREDICTIONS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    dated = config.PREDICTIONS_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    dated.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote {config.PREDICTIONS_JSON}")
    print(f"wrote {dated}")


if __name__ == "__main__":
    main()
