# Step 01 — Project Setup

**Week:** 1 | **Est. time:** 2–4h | **Depends on:** nothing | **Blocks:** [02-data-and-eda](02-data-and-eda.md)

Scaffold the repository, dependency management, environment configuration, and Kaggle data download.

---

## Objectives

- Git repo initialized in `bgg/` (not parent directory)
- Installable Python package with dependency groups
- Environment variables documented and loadable
- BGG raw data downloaded to `data/raw/`

---

## Tasks

### Task 1: Initialize repository

**Subtasks:**

1. `cd` into `bgg/` and run `git init`
2. Create directory skeleton:
   ```
   data/raw/ data/processed/ data/models/ data/user_profiles/
   notebooks/ src/bgg/ frontend/ scripts/ docker/ tests/
   rulebooks/ chroma_data/ mlruns/
   ```
3. Add `rulebooks/.gitkeep` and `data/raw/.gitkeep`
4. Create `.gitignore` (see [folder-structure.md](../architecture/folder-structure.md))

**Checkpoint:**

- [ ] `git status` shows clean structure; no repo at parent `C:\Users\josec`

---

### Task 2: Create `pyproject.toml`

**Subtasks:**

1. Define package `bgg` with `src` layout (`[tool.setuptools.packages.find] where = ["src"]`)
2. Create optional dependency groups:

| Group | Packages |
|-------|----------|
| `core` | `fastapi`, `uvicorn[standard]`, `pydantic-settings`, `httpx`, `pandas`, `pyarrow` |
| `spark` | `pyspark` |
| `rag` | `chromadb`, `langchain`, `langchain-community`, `sentence-transformers`, `pymupdf` |
| `dev` | `pytest`, `pytest-mock`, `ruff`, `jupyter`, `ipykernel` |

3. Pin major versions for reproducibility (e.g. `pyspark>=3.5,<4`, `mlflow>=2.14,<3`)
4. Add `mlflow`, `streamlit` to core or separate `ml` group
5. Install: `pip install -e ".[dev,spark,rag]"`

**Starter `pyproject.toml` snippet:**

```toml
[project]
name = "bgg-assistant"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "pydantic-settings>=2.3",
    "mlflow>=2.14,<3",
    "streamlit>=1.35",
    "httpx>=0.27",
    "pandas>=2.2",
    "pyarrow>=16.0",
]

[project.optional-dependencies]
spark = ["pyspark>=3.5,<4"]
rag = [
    "chromadb>=0.5",
    "langchain>=0.2",
    "langchain-community>=0.2",
    "sentence-transformers>=3.0",
    "pymupdf>=1.24",
]
dev = ["pytest>=8.0", "pytest-mock>=3.14", "ruff>=0.4", "jupyter>=1.0", "ipykernel>=6.29"]

[build-system]
requires = ["setuptools>=69"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]
```

**Checkpoint:**

- [ ] `pip install -e ".[dev,spark,rag]"` succeeds without errors

---

### Task 3: Environment configuration

**Subtasks:**

1. Create `.env.example`:

```bash
# Paths
BGG_DATA_DIR=./data
RULEBOOKS_DIR=./rulebooks

# Services
MLFLOW_TRACKING_URI=http://localhost:5000
CHROMA_HOST=http://localhost:8001
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=phi3:mini

# MLflow model
MLFLOW_MODEL_NAME=bgg-recommender
MLFLOW_MODEL_STAGE=Production

# Spark (training)
SPARK_MASTER=local[*]

# Frontend
FASTAPI_URL=http://localhost:8000
```

2. Copy to `.env` for local dev (gitignored)
3. Implement `src/bgg/config.py`:

```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    bgg_data_dir: Path = Path("./data")
    rulebooks_dir: Path = Path("./rulebooks")
    mlflow_tracking_uri: str = "http://localhost:5000"
    chroma_host: str = "http://localhost:8001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"
    mlflow_model_name: str = "bgg-recommender"
    mlflow_model_stage: str = "Production"
    spark_master: str = "local[*]"
    fastapi_url: str = "http://localhost:8000"

    @property
    def processed_dir(self) -> Path:
        return self.bgg_data_dir / "processed"

    @property
    def raw_dir(self) -> Path:
        return self.bgg_data_dir / "raw"

settings = Settings()
```

4. Add `src/bgg/__init__.py` (empty or version string)

**Checkpoint:**

- [ ] `python -c "from bgg.config import settings; print(settings.processed_dir)"` prints path

---

### Task 4: Kaggle data acquisition

**Subtasks:**

1. Create Kaggle API token (`~/.kaggle/kaggle.json`) if not present
2. Identify dataset — recommended: **[mshepherd/board-games](https://www.kaggle.com/datasets/mshepherd/board-games)** (Recommend.Games scraper subset) or similar with:
   - `games.csv` (or equivalent): `id`, `name`, `minplayers`, `maxplayers`, `playingtime`, `averageweight`, ratings
   - `mechanics.csv` / `categories.csv` or joined columns
   - `ratings.csv` or user ratings for ALS
3. Implement `scripts/download_kaggle_data.py`:

```python
"""Download BGG dataset from Kaggle into data/raw/."""
import subprocess
import sys
from pathlib import Path

DATASET = "mshepherd/board-games"  # adjust if using different dataset
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW_DIR), "--unzip"],
        check=True,
    )
    print(f"Downloaded to {RAW_DIR}")

if __name__ == "__main__":
    main()
```

4. Document actual CSV filenames in this file after first download (dataset schemas vary)
5. Run: `python scripts/download_kaggle_data.py`

**Checkpoint:**

- [ ] At least one games CSV and ratings/mechanics data present under `data/raw/`
- [ ] File list documented in a comment at top of `download_kaggle_data.py`

---

### Task 5: Initial package stubs

**Subtasks:**

1. Create empty `__init__.py` in `src/bgg/recommender/`, `src/bgg/rag/`, `src/bgg/api/`
2. Add `tests/conftest.py` with `settings` fixture
3. Run `ruff check src/` (should pass on stubs)

**Checkpoint:**

- [ ] Package imports work: `python -c "import bgg"`

---

## Definition of Done

All checkpoints above are checked. Mark Step 01 complete in [docs/README.md](../README.md) progress table.

**Next:** [02-data-and-eda](02-data-and-eda.md)

---

## Reference

- [Folder structure](../architecture/folder-structure.md)
- [Pitfalls — Spark on Pi](../architecture/pitfalls-and-mitigations.md#2-pyspark-on-raspberry-pi)
