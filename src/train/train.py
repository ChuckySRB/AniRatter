"""Step 3 — Train candidate models, pick the best by CV MAE, refit on full data, save.

Reads:  data/processed/features_train.parquet, feature_spec.json
Writes: data/models/best_model.joblib, data/models/metrics.json

Run: python -m src.train.train
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split

import config
from src.preprocess.feature_builder import load_spec

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
    LGBM_VERSION = lgb.__version__
except ImportError:
    LGBM_AVAILABLE = False
    LGBM_VERSION = None


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=config.ROOT_DIR, text=True
        ).strip()
    except Exception:
        return "unknown"


def _make_candidates() -> list[tuple[str, object, dict]]:
    """Returns (name, estimator, param_grid) tuples."""
    candidates: list[tuple[str, object, dict]] = [
        (
            "RandomForestRegressor",
            RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 16],
                "min_samples_leaf": [1, 3],
            },
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(random_state=config.RANDOM_STATE),
            {
                "n_estimators": [200, 400],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
    ]
    if LGBM_AVAILABLE:
        candidates.append((
            "LGBMRegressor",
            lgb.LGBMRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbose=-1),
            {
                "n_estimators": [300, 600],
                "learning_rate": [0.05, 0.1],
                "num_leaves": [31, 63],
            },
        ))
    return candidates


def main() -> None:
    if not config.FEATURES_PARQUET.exists():
        raise SystemExit(f"missing {config.FEATURES_PARQUET} — run `python -m src.preprocess.build_features` first")

    spec = load_spec(config.FEATURE_SPEC_JSON, config.FEATURE_SPEC_JOBLIB)
    df = pd.read_parquet(config.FEATURES_PARQUET)
    y = df["score"].values
    X = df[spec.column_order].values
    print(f"training on {X.shape[0]} rows × {X.shape[1]} features")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    candidates = _make_candidates()
    cv_results: list[dict] = []
    best_name = None
    best_estimator = None
    best_params = None
    best_cv_mae = math.inf

    for name, estimator, grid in candidates:
        print(f"\n=== {name} ===")
        gs = GridSearchCV(
            estimator,
            grid,
            scoring="neg_mean_absolute_error",
            cv=config.CV_FOLDS,
            n_jobs=-1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        cv_mae = -gs.best_score_
        print(f"  best CV MAE: {cv_mae:.3f}")
        print(f"  best params: {gs.best_params_}")
        cv_results.append({
            "model": name,
            "best_cv_mae": cv_mae,
            "best_params": gs.best_params_,
        })
        if cv_mae < best_cv_mae:
            best_cv_mae = cv_mae
            best_name = name
            best_estimator = gs.best_estimator_
            best_params = gs.best_params_

    print(f"\n*** best model: {best_name} (CV MAE {best_cv_mae:.3f}) ***")

    # Honest holdout metrics
    test_pred = best_estimator.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, test_pred))
    test_rmse = float(math.sqrt(mean_squared_error(y_test, test_pred)))
    test_r2 = float(r2_score(y_test, test_pred))
    print(f"holdout MAE: {test_mae:.3f} | RMSE: {test_rmse:.3f} | R2: {test_r2:.3f}")

    # Refit on 100% of the data (we want the best possible final model — holdout was for honest reporting)
    print("refitting on full dataset…")
    final_model = best_estimator.__class__(**best_estimator.get_params())
    final_model.fit(X, y)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, config.MODEL_JOBLIB)

    metrics = {
        "model_name": best_name,
        "best_params": best_params,
        "cv_mae": best_cv_mae,
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "n_train": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "candidates": cv_results,
        "spec_hash": spec.spec_hash,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": LGBM_VERSION,
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    config.METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    print(f"wrote {config.MODEL_JOBLIB}")
    print(f"wrote {config.METRICS_JSON}")


if __name__ == "__main__":
    main()
