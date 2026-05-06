"""Pipeline configuration. Module-level constants — import what you need."""
from pathlib import Path

# AniList user
USER = "ChuckySRB"

# Paths (relative to repo root)
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"
PREDICTIONS_DIR = DATA_DIR / "predictions"
WEB_DIR = ROOT_DIR / "web"

# Raw file names
RAW_RATED_ANIME = RAW_DIR / "rated_anime.json"
RAW_RATED_MANGA = RAW_DIR / "rated_manga.json"
RAW_PLANNING_ANIME = RAW_DIR / "planning_anime.json"
RAW_PLANNING_MANGA = RAW_DIR / "planning_manga.json"

# Processed artifact paths
FEATURES_PARQUET = PROCESSED_DIR / "features_train.parquet"
FEATURE_SPEC_JSON = PROCESSED_DIR / "feature_spec.json"
FEATURE_SPEC_JOBLIB = PROCESSED_DIR / "feature_spec.joblib"

# Model artifact paths
MODEL_JOBLIB = MODELS_DIR / "best_model.joblib"
METRICS_JSON = MODELS_DIR / "metrics.json"

# Predictions artifact paths
PREDICTIONS_JSON = PREDICTIONS_DIR / "predictions.json"

# Preprocessing
MIN_TAG_RANK = 50

# Training
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Recommendation score: rec = clip( (P - REC_BASE_OFFSET) + REC_PERSONAL_WEIGHT * (P - meanScore), -REC_CLIP, REC_CLIP )
REC_PERSONAL_WEIGHT = 1.5
REC_BASE_OFFSET = 50
REC_CLIP = 100
