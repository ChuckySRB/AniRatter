# 🎌 AniRatter

> Predict how much you'll like the anime & manga sitting on your plan-to-watch list — trained on your own AniList ratings.

AniRatter pulls your AniList scores, trains a model on what you've already rated, and uses it to predict scores for everything in your plan-to-watch / plan-to-read list. Predictions are blended with the AniList community average into a single **recommendation score** in the range `[-100, +100]`, then surfaced through a clean static web page with cover art and filters.

📺 ➡ 🤖 ➡ 📊 ➡ 🔮

---

## ✨ Features

- 🔁 **5 independent pipeline steps** — fetch, preprocess, train, predict, build site. Re-run any step in isolation; intermediate artifacts are persisted.
- 🧠 **Model bake-off** — RandomForest, GradientBoosting, and LightGBM are tuned via 5-fold cross-validation; the best one wins.
- 🎯 **Personal `rec_score`** — combines "how much I'll like it" with "how much I'll like it more than the crowd".
- 🖼️ **Cover-art gallery** — vanilla-JS static page, no build step, opens straight from disk.
- 🧹 **Deduped & validated** — duplicates removed by `media.id`, response shape checked, schema mismatch protections at predict time.
- 🛠️ **Reproducible** — sklearn version, spec hash, git commit, and timestamps are all baked into the saved metrics.

---

## 📂 Project layout

```
AniRatter/
├── 📄 config.py                       # username, paths, knobs
├── 📄 requirements.txt
├── 🗂️ src/
│   ├── 🔌 anilist/    queries.py · client.py
│   ├── 📥 data/       fetch.py
│   ├── 🧪 preprocess/ feature_builder.py · build_features.py
│   ├── 🌲 train/      train.py
│   ├── 🔮 predict/    predict.py
│   └── 🌐 web/        build_site.py · serve.py
├── 🌐 web/             index.html · styles.css · app.js
└── 🗄️ data/            (gitignored)
    ├── raw/        # 4 × json — fresh API dumps
    ├── processed/  # features.parquet + feature_spec
    ├── models/     # best_model.joblib + metrics.json
    └── predictions/# predictions.json + dated copies
```

---

## 🚀 Quick start

### 1️⃣ Install

```bash
pip install -r requirements.txt
```

### 2️⃣ Tell it who you are

Open `config.py` and set your AniList username:

```python
USER = "YourAniListName"
```

### 3️⃣ Run the pipeline

```bash
python -m src.data.fetch                  # 📥 pull raw data
python -m src.preprocess.build_features   # 🧪 build features
python -m src.train.train                 # 🌲 train + pick best model
python -m src.predict.predict             # 🔮 score your plan-to-watch
python -m src.web.build_site              # 🌐 stage the page
python -m src.web.serve                   # 🚪 → http://localhost:8000
```

That's it — open the printed URL and browse your predictions. 🎉

> 💡 You can re-run any single step. Re-trained the model? Just rerun `predict` + `build_site`. New rating on AniList? Rerun the whole chain — each step takes seconds.

---

## 🔬 How it works

### 📥 Step 1 — Fetch (`src/data/fetch.py`)

Hits the AniList GraphQL API for 4 segments: rated anime, rated manga, planning anime, planning manga. Pulls media metadata (title, format, year, source, genres, tags + rank, meanScore, popularity, favourites, cover image, episodes/chapters, site URL). Dedupes by `media.id`, retries on 429, validates response shape.

### 🧪 Step 2 — Preprocess (`src/preprocess/build_features.py`)

A `FeatureBuilder` class with `.fit() / .transform() / .fit_transform()` mirroring sklearn's API. The fitted state — column order, vocabularies, scaler stats, train medians — is serialized to `feature_spec.json` (+ joblib) so prediction time uses the **exact same transform**.

| 📊 Feature | Treatment |
|---|---|
| meanScore, averageScore, popularity, favourites, seasonYear, mean_top_tag_rank | `StandardScaler` + median fill |
| `is_manga` | binary (replaces `type` one-hot) |
| `format`, `source` | one-hot with always-present `_other` bucket |
| `genres` | multi-label binary, vocab persisted |
| `tags` (rank ≥ 50) | multi-label binary; rank captured via aggregate `mean_top_tag_rank` |

Filters training rows to `0 < score ≤ 100` and `meanScore is not null`.

### 🌲 Step 3 — Train (`src/train/train.py`)

Holds out 20% as a final test set, then on the 80% runs **5-fold `GridSearchCV` with neg-MAE** across:

- 🌳 RandomForestRegressor — `n_estimators ∈ {200, 400}`, `max_depth ∈ {None, 16}`, `min_samples_leaf ∈ {1, 3}`
- 🚀 GradientBoostingRegressor — `n_estimators ∈ {200, 400}`, `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {3, 5}`
- ⚡ LGBMRegressor — `n_estimators ∈ {300, 600}`, `learning_rate ∈ {0.05, 0.1}`, `num_leaves ∈ {31, 63}`

The model with the lowest mean CV MAE wins. Test metrics (MAE / RMSE / R²) are reported on the holdout, then the final saved model is **refit on 100% of the data**. `metrics.json` records: chosen model + params, all CV results, sklearn version, spec hash, git commit, timestamp.

### 🔮 Step 4 — Predict (`src/predict/predict.py`)

Loads model + spec, asserts the spec hash matches what the model was trained with, applies the same transform to planning items, predicts scores, and computes:

```
rec_score = clip( (P − 50) + 1.5 × (P − meanScore),  −100,  +100 )
```

Where:
- 🎯 `(P − 50)` — **absolute appeal**: how high will you rate it on its own merit
- 🔥 `1.5 × (P − meanScore)` — **personal edge**: how much you'll like it more than the crowd, weighted higher because the differential is the more actionable signal
- ✂️ clipped so extreme outliers don't blow out the scale

Examples:

| Predicted | Mean | rec_score | vibe |
|---|---|---|---|
| 98 | 89 | **+61.5** | 🌟 love it & beat the crowd |
| 85 | 85 | **+35.0** | 👍 solidly good, no edge |
| 70 | 80 | **+5.0**  | 🤷 decent, but crowd likes more |
| 55 | 80 | **−32.5** | 😐 mid for me, worse than crowd |
| 40 | 75 | **−62.5** | ⏭️ skip |

Logs the **clip fraction**: % of items hitting `±100`. If it's >20%, the formula weights need revisiting.

### 🌐 Step 5 — Build site (`src/web/build_site.py`)

Copies `predictions.json` into `web/`, AND inlines it into `index.html` as a `<script type="application/json">` block — so the page works opened directly via `file://` (browsers block `fetch()` over file://). For a multi-MB JSON, prefer the local server (`python -m src.web.serve`).

---

## 🌐 The web page

A single static page, vanilla JS, no framework, no build step. ✨

- 📑 **Anime / Manga tabs** with item counts
- 🔃 **Sort** by Recommendation, Predicted, Mean, or Year
- 🎭 **Genre multi-select chips** — pick any combination
- 🎬 **Format dropdown**, year range, title search
- 🃏 **Card grid** — cover, title, year · format · ep/ch, genre chips, three big numbers (Predicted / Mean / Rec)
- 🟢🔴 `rec_score` color-coded by sign
- 🔗 click a card → opens it on AniList
- 🌗 **Light/dark** via `prefers-color-scheme`
- 📱 mobile-responsive

---

## ⚙️ Configuration

Everything lives in `config.py`:

```python
USER = "ChuckySRB"

MIN_TAG_RANK = 50          # tag confidence threshold (0–100)
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

REC_PERSONAL_WEIGHT = 1.5  # weight on (P − meanScore)
REC_BASE_OFFSET = 50       # P − offset = absolute appeal
REC_CLIP = 100
```

Want a more aggressive "find me hidden gems" recommender? Bump `REC_PERSONAL_WEIGHT`. Want a more "popular & quality first" one? Lower it.

---

## 📦 Dependencies

```
pandas>=2.0
scikit-learn==1.5.*    # pinned: pickled estimators break across minor versions
lightgbm>=4.0
requests>=2.31
joblib>=1.3
pyarrow>=14
```

---

## 📜 Version history

This is a rewrite of the original 5-file, ~314-LOC version. The old files (`app.py`, `machine_learing.py`, `anilist_queries.py`, `anilist_requests.py`, `predictions/*.csv`) are kept around until you're happy with the new pipeline — feel free to delete them once verified.

What changed in the rewrite:
- 🪄 5 independent pipeline steps (was: one monolithic script)
- 💾 model + preprocessor are now persisted (was: retrained every run)
- 🧮 model bake-off with CV (was: default-config RandomForest)
- ➕ now uses popularity, favourites, averageScore, mean_top_tag_rank (were fetched but ignored)
- 🧹 dedupes API duplicates (was: silently kept)
- 🛡️ unseen-category & unseen-genre/tag handling at predict time
- 📊 principled `rec_score` (was: a custom Excel formula called "ChuckySCORE")
- 🌐 full static web page with cover art (was: open the CSV in Excel and squint)

---

## 🤝 Credits

Data from [AniList](https://anilist.co) (no auth needed for public lists). Made for fun. 💙
