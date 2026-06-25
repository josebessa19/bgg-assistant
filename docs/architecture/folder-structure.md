# Repository Folder Structure

Target layout for the BGG Board Game Assistant. Create directories incrementally as each implementation step requires them.

```
bgg/
├── README.md                          # Portfolio entry point → links to docs/
├── docker-compose.yml                 # Full stack (Step 06)
├── .env.example                       # Environment variable template (Step 01)
├── pyproject.toml                     # Dependencies: core, spark, rag, dev groups (Step 01)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                     # Lint, test, docker build (Step 07)
│
├── data/
│   ├── raw/                           # Kaggle BGG CSVs (gitignored)
│   ├── processed/                     # Parquet outputs (gitignored)
│   │   ├── games_features.parquet
│   │   └── ratings_implicit.parquet
│   ├── models/                        # Exported inference artifacts for Pi (gitignored)
│   └── user_profiles/                 # JSON liked-game lists (volume mount)
│
├── notebooks/
│   ├── 01_eda.ipynb                   # PySpark EDA (Step 02)
│   └── 02_recommender.ipynb           # ALS + content training (Step 03)
│
├── src/bgg/
│   ├── __init__.py
│   ├── config.py                      # pydantic-settings: paths, URLs, Spark config
│   ├── recommender/
│   │   ├── __init__.py
│   │   ├── features.py                # Mechanics/categories encoding, numerics normalization
│   │   ├── content_filter.py          # Cosine similarity + hard session filters
│   │   ├── als_train.py               # Spark MLlib ALS training utilities
│   │   └── inference.py               # Hybrid ranker + cold-start fallback
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py                # PyMuPDF → ~500-token chunks
│   │   ├── rag_pipeline.py            # Embed, index, retrieve, generate
│   │   └── prompts.py                 # LangChain prompt templates
│   └── api/
│       ├── __init__.py
│       ├── main.py                    # FastAPI app, health, router mount
│       ├── recommender_api.py         # POST /recommend
│       └── chatbot_api.py             # POST /ask
│
├── frontend/
│   └── streamlit_app.py               # Tabs: Recommender | Rulebook Chat (Step 05)
│
├── scripts/
│   ├── download_kaggle_data.py        # Fetch BGG dataset from Kaggle (Step 01)
│   ├── preprocess_to_parquet.py       # Spark batch preprocessing (Step 02)
│   ├── train_recommender.py           # CLI training wrapper (Step 03)
│   └── index_rulebooks.py             # Batch + incremental Chroma indexing (Step 04)
│
├── docker/
│   ├── Dockerfile.api                 # FastAPI + ML/RAG deps (Step 06)
│   ├── Dockerfile.streamlit           # Streamlit frontend (Step 06)
│   └── Dockerfile.mlflow              # Optional thin MLflow wrapper
│
├── rulebooks/                         # PDF rulebooks (bind mount, not in git)
│   └── game_registry.json             # slug ↔ filename ↔ optional BGG ID mapping
├── chroma_data/                       # Chroma persistence (gitignored)
├── mlruns/                            # MLflow artifacts + sqlite backend (gitignored)
│
└── tests/
    ├── conftest.py                    # Fixtures, mocks for Ollama/Chroma
    ├── test_recommender.py
    ├── test_rag_chunking.py
    └── test_api.py
```

## File Responsibilities

### Configuration

| File | Responsibility |
|------|----------------|
| `pyproject.toml` | Single source of truth for Python deps and optional groups (`core`, `spark`, `rag`, `dev`) |
| `.env.example` | Documents all env vars; copy to `.env` locally |
| `src/bgg/config.py` | Loads settings via `pydantic-settings`; used by API, scripts, notebooks |

### Data Layer

| Path | Responsibility |
|------|----------------|
| `data/raw/` | Immutable Kaggle downloads |
| `data/processed/` | Clean Parquet consumed by recommender training and API lookup |
| `data/models/` | Lightweight inference artifacts exported for Raspberry Pi (no Spark at runtime) |
| `data/user_profiles/` | Optional persisted user liked-game lists |

### Recommender (`src/bgg/recommender/`)

| File | Responsibility |
|------|----------------|
| `features.py` | Build feature vectors: multi-hot mechanics/categories, normalized player count and duration |
| `content_filter.py` | Apply hard filters (players, duration, coop/competitive); cosine rank vs user profile |
| `als_train.py` | Spark ALS on implicit BGG ratings; export factor matrices |
| `inference.py` | Hybrid scoring, cold-start fallback, exclude already-liked games |

### RAG (`src/bgg/rag/`)

| File | Responsibility |
|------|----------------|
| `chunking.py` | Extract text from PDFs, split into chunks with metadata |
| `rag_pipeline.py` | Chroma collection management, retrieval, LangChain → Ollama chain |
| `prompts.py` | System/user prompts for grounded rulebook Q&A |

### API (`src/bgg/api/`)

| File | Responsibility |
|------|----------------|
| `main.py` | App factory, CORS, lifespan (load models), `/health` |
| `recommender_api.py` | `POST /recommend` — validate IDs, call `inference.hybrid_recommend` |
| `chatbot_api.py` | `POST /ask` — retrieve chunks, generate answer with sources |

### Frontend

| File | Responsibility |
|------|----------------|
| `frontend/streamlit_app.py` | Two tabs; calls FastAPI via `FASTAPI_URL`; game search multiselect |

### Scripts

| Script | When to run |
|--------|-------------|
| `download_kaggle_data.py` | Once after Kaggle API setup |
| `preprocess_to_parquet.py` | After raw data download or when schema changes |
| `train_recommender.py` | After preprocessing; logs to MLflow |
| `index_rulebooks.py` | On deploy, when PDFs added/changed; supports incremental mode |

### Docker

| File | Responsibility |
|------|----------------|
| `docker-compose.yml` | ollama, chromadb, mlflow, api, streamlit |
| `docker/Dockerfile.api` | Production API image |
| `docker/Dockerfile.streamlit` | Lightweight Streamlit image |

## Gitignore Essentials

```
data/raw/
data/processed/
data/models/
chroma_data/
mlruns/
.env
__pycache__/
*.pyc
.ipynb_checkpoints/
rulebooks/*.pdf
```

Keep `rulebooks/game_registry.json` and `rulebooks/.gitkeep` in git; exclude actual PDFs.

## Environment Variables (cross-reference)

See `.env.example` (created in Step 01). Key vars:

- `BGG_DATA_DIR` — path to `data/`
- `RULEBOOKS_DIR` — path to PDF folder
- `CHROMA_HOST` — Chroma HTTP URL
- `OLLAMA_BASE_URL` — Ollama API URL
- `MLFLOW_TRACKING_URI` — MLflow server URL
- `SPARK_MASTER` — `local[*]` for dev training
- `OLLAMA_MODEL` — `phi3:mini` (Pi) or `mistral:7b` (dev)
- `FASTAPI_URL` — used by Streamlit container
