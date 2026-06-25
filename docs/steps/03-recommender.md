# Step 03 — Recommender

**Week:** 2 | **Est. time:** 5–10h | **Depends on:** [02-data-and-eda](02-data-and-eda.md) | **Blocks:** [05-api-and-frontend](05-api-and-frontend.md)

Train a hybrid recommender (Spark MLlib ALS + content-based filtering), track experiments in MLflow, and build production inference code.

---

## Objectives

- ALS model trained on implicit BGG ratings
- Content-based index from game features
- Hybrid ranker with session filters and cold-start fallback
- MLflow experiment with logged metrics and registered model
- CLI inference returns sensible 2-player recommendations in &lt; 5s

---

## Algorithm Specification

### Collaborative (ALS)

| Parameter | Value |
|-----------|-------|
| Input | `ratings_implicit.parquet` → `(user_id, game_id, implicit_strength)` |
| Algorithm | Spark MLlib `ALS` |
| `rank` | 50 |
| `maxIter` | 10 |
| `regParam` | 0.1 |
| `implicitPrefs` | `true` |
| `coldStartStrategy` | `drop` |
| Eval | Holdout 20% user-game pairs; RMSE on explicit scale; Precision@10 |

### Content-based

| Component | Detail |
|-----------|--------|
| Features | Multi-hot `mechanics` + `categories`; normalized `avg_weight`, `playing_time`, `bayes_average` |
| User profile | Mean vector of liked games' feature vectors |
| Similarity | Cosine similarity vs all candidate games |
| Hard filters | Applied before ranking (see below) |

### Session filters (hard)

| Filter | Rule |
|--------|------|
| `players` | `min_players <= players <= max_players` |
| `max_duration_minutes` | `playing_time <= max_duration` (or use `max_play_time`) |
| `mode` | `cooperative` → `is_cooperative == true`; `competitive` → `false`; `any` → no filter |

### Hybrid scoring

```
if len(liked_game_ids) >= 3:
    score = 0.6 * als_score_normalized + 0.4 * content_score
else:
  # cold start
    score = 0.7 * content_score + 0.3 * popularity_normalized
```

- Exclude games already in `liked_game_ids`
- `popularity_normalized` = min-max scaled `bayes_average`
- Tune weights via MLflow params

---

## Tasks

### Task 1: Feature engineering module

**Subtasks:**

1. Create `src/bgg/recommender/features.py`:

```python
"""Feature encoding for content-based filtering."""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.ml.feature import CountVectorizer, Normalizer, VectorAssembler

MECHANICS_COL = "mechanics_vec"
CATEGORIES_COL = "categories_vec"
FEATURE_COL = "features"

def encode_game_features(games_df: DataFrame) -> DataFrame:
    """Build combined feature vector per game."""
    # Tokenize array columns → CountVectorizer → assemble with numerics → L2 normalize
    ...
```

2. Functions:
   - `build_mechanics_vocab(games_df)` → vocabulary list
   - `encode_games(games_df, vocab)` → DataFrame with `features` vector column
   - `user_profile_vector(liked_ids, encoded_df)` → mean vector (pandas/numpy for inference)

3. Persist vocabulary + scaler params to JSON in `data/models/feature_config.json`

**Checkpoint:**

- [ ] Feature vector dimension documented (e.g. ~200–500)
- [ ] Sample game pair similarity is intuitive (similar mechanics → higher cosine)

---

### Task 2: Content-based filter

**Subtasks:**

1. Create `src/bgg/recommender/content_filter.py`:

```python
def apply_session_filters(df, players: int, max_duration: int, mode: str):
    """Hard filter before ranking."""
    ...

def rank_by_content(candidates_df, user_vector, top_k: int) -> list[dict]:
    """Cosine similarity ranking."""
    ...
```

2. Return list of `{game_id, name, content_score}`

**Checkpoint:**

- [ ] Filter `players=2, max_duration=90` reduces candidate set sensibly
- [ ] Top result for Wingspan-like input includes engine-building games

---

### Task 3: ALS training

**Subtasks:**

1. Create `src/bgg/recommender/als_train.py`:

```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

def train_als(ratings_df, rank=50, max_iter=10, reg_param=0.1) -> ALSModel:
    train, test = ratings_df.randomSplit([0.8, 0.2], seed=42)
    als = ALS(
        rank=rank, maxIter=max_iter, regParam=reg_param,
        userCol="user_id", itemCol="game_id", ratingCol="implicit_strength",
        implicitPrefs=True, coldStartStrategy="drop",
    )
    model = als.fit(train)
    predictions = model.transform(test)
    # evaluate ...
    return model
```

2. Export item factors for inference without Spark:
   - `model.itemFactors.toPandas().to_parquet("data/models/item_factors.parquet")`
   - Or save full Spark model to MLflow

3. Create `notebooks/02_recommender.ipynb` — mirror training with eval plots

**Checkpoint:**

- [ ] Holdout RMSE logged
- [ ] Precision@10 computed on sample users

---

### Task 4: Hybrid inference

**Subtasks:**

1. Create `src/bgg/recommender/inference.py`:

```python
COLD_START_THRESHOLD = 3

def hybrid_recommend(
    liked_ids: list[int],
    players: int,
    max_duration: int,
    mode: str,
    top_k: int = 10,
) -> list[dict]:
    """
    Returns [{game_id, name, score, reason, als_score, content_score}]
    """
    candidates = load_games_features()
    candidates = apply_session_filters(candidates, players, max_duration, mode)
    candidates = candidates[~candidates.game_id.isin(liked_ids)]

    if len(liked_ids) >= COLD_START_THRESHOLD:
        als_scores = score_als(liked_ids, candidates)
        content_scores = score_content(liked_ids, candidates)
        # combine 0.6 / 0.4
    else:
        content_scores = score_content(liked_ids, candidates)
        popularity = normalize(candidates["bayes_average"])
        # combine 0.7 / 0.3

    return top_k_results
```

2. `reason` field: human-readable e.g. `"Similar mechanics to liked games; strong 2-player rating"`

**Checkpoint:**

- [ ] 5 liked games + `players=2` → 10 results in &lt; 5s
- [ ] 1 liked game → cold-start path works, no crash

---

### Task 5: MLflow integration

**Subtasks:**

1. Create `scripts/train_recommender.py`:

```python
import mlflow
from bgg.config import settings

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

with mlflow.start_run(run_name="als-content-hybrid"):
    mlflow.log_params({"rank": 50, "max_iter": 10, "hybrid_als_weight": 0.6})
    model = train_als(...)
    metrics = evaluate(...)
    mlflow.log_metrics(metrics)
    mlflow.spark.log_model(model, "als_model")
    mlflow.log_artifact("data/models/feature_config.json")
    mlflow.log_artifact("data/models/content_vectors.parquet")
```

2. Register model: `mlflow.register_model("runs:/<run_id>/als_model", "bgg-recommender")`
3. Transition to `Production` stage after manual review

**MLflow artifacts:**

| Artifact | Purpose |
|----------|---------|
| `als_model/` | Spark ALS model |
| `feature_config.json` | Vocab + normalization params |
| `content_vectors.parquet` | Precomputed game vectors |
| `item_factors.parquet` | Lightweight ALS inference |
| `params.json` | Hybrid weights |

**Checkpoint:**

- [ ] MLflow UI shows ≥ 1 run with params and metrics
- [ ] Model registered as `bgg-recommender`

---

### Task 6: Export for Raspberry Pi

**Subtasks:**

1. Script or notebook cell to copy to `data/models/`:
   - `item_factors.parquet`
   - `content_vectors.parquet`
   - `feature_config.json`
   - `game_lookup.parquet` (id, name, filters)
2. Verify `inference.py` loads from `data/models/` without Spark

**Checkpoint:**

- [ ] `python -c "from bgg.recommender.inference import hybrid_recommend; ..."` works with exported artifacts only

---

## Notebook Outline (`02_recommender.ipynb`)

| Section | Content |
|---------|---------|
| 1. Load Parquet | games + ratings |
| 2. Feature encoding | vocab, vectors |
| 3. ALS train | hyperparams, eval |
| 4. Hybrid demo | liked games input, 2-player filter |
| 5. Cold start | 1-game input |
| 6. MLflow log | register model |
| 7. Export | Pi artifacts |

---

## CLI Test

```bash
python -c "
from bgg.recommender.inference import hybrid_recommend
results = hybrid_recommend(
    liked_ids=[174430, 161936, 182028, 220308, 167791],
    players=2,
    max_duration=90,
    mode='any',
    top_k=10,
)
for r in results:
    print(r['name'], round(r['score'], 3), r['reason'])
"
```

---

## Definition of Done

- [ ] ALS + content model in MLflow
- [ ] Offline eval metrics logged
- [ ] CLI returns 10 sensible 2-player games in &lt; 5s
- [ ] Cold-start path tested with 1 liked game
- [ ] Pi export artifacts in `data/models/`

Mark Step 03 complete in [docs/README.md](../README.md).

**Next:** [04-rag-pipeline](04-rag-pipeline.md)

---

## Reference

- [Pitfalls — ALS cold start](../architecture/pitfalls-and-mitigations.md#3-als-cold-start-few-user-ratings)
- [Pitfalls — Spark on Pi](../architecture/pitfalls-and-mitigations.md#2-pyspark-on-raspberry-pi)
