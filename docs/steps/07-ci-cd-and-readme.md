# Step 07 — CI/CD & README

**Week:** 4 | **Est. time:** 2–4h | **Depends on:** [06-docker-and-deploy](06-docker-and-deploy.md) | **Blocks:** nothing (final step)

Set up GitHub Actions CI, write portfolio README, and complete release checklist.

---

## Objectives

- CI pipeline: lint → test → docker build
- Portfolio-ready `README.md` with badges, architecture, screenshots
- Release checklist completed

---

## Tasks

### Task 1: Ruff configuration

**Subtasks:**

1. Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

2. Run locally: `ruff check src/ tests/ scripts/`
3. Fix any issues

**Checkpoint:**

- [ ] `ruff check` exits 0

---

### Task 2: Pytest configuration

**Subtasks:**

1. Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
    "integration: tests requiring Ollama/Chroma (deselect in CI)",
]
```

2. Ensure unit tests mock external services (Ollama, Chroma HTTP)
3. Run: `pytest -m "not integration"`

**Checkpoint:**

- [ ] `pytest -m "not integration"` passes locally

---

### Task 3: GitHub Actions workflow

**Subtasks:**

1. Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install -e ".[dev,rag]"

      - name: Ruff
        run: ruff check src/ tests/ scripts/

      - name: Pytest
        run: pytest -m "not integration" -v

  docker-build:
    runs-on: ubuntu-latest
    needs: lint-and-test
    steps:
      - uses: actions/checkout@v4

      - name: Build API image
        run: docker build -f docker/Dockerfile.api -t bgg-api .

      - name: Build Streamlit image
        run: docker build -f docker/Dockerfile.streamlit -t bgg-streamlit .
```

2. Push to GitHub; verify Actions tab

**Notes:**

- No Ollama/Chroma in CI — mock RAG and recommender in tests
- Docker build does not run `docker compose up` (no integration deploy in CI)
- Optional: add `paths-ignore` for `docs/**` only changes

**Checkpoint:**

- [ ] CI green on default branch

---

### Task 4: Portfolio README

**Subtasks:**

1. Write root `README.md` with sections below
2. Add badges (replace `YOUR_USER`):

```markdown
[![CI](https://github.com/YOUR_USER/bgg/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USER/bgg/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Docker](https://img.shields.io/badge/docker-compose-blue)
![MLflow](https://img.shields.io/badge/MLflow-tracking-green)
```

**README structure:**

| # | Section | Content |
|---|---------|---------|
| 1 | Title + one-liner | "Self-hosted board game recommender and rulebook RAG chatbot using BGG data" |
| 2 | Demo | GIF or 2–3 screenshots (see checklist below) |
| 3 | Features | Bullet list: hybrid recommender, 2-player filters, local RAG, MLflow, Docker |
| 4 | Architecture | Mermaid diagram (copy from [docs/README.md](../README.md)) |
| 5 | Tech stack | Table matching project spec |
| 6 | Quick start — local | venv, install, download data, preprocess, train, run API + Streamlit |
| 7 | Quick start — Docker | `docker compose up`, pull model, index rulebooks |
| 8 | ML experiments | Screenshot of MLflow metrics; mention ALS + content hybrid |
| 9 | Project structure | Link to [folder-structure.md](../architecture/folder-structure.md) |
| 10 | Implementation guide | Link to [docs/README.md](../README.md) |
| 11 | Limitations | Pi latency, no user accounts, English rulebooks only |
| 12 | License | MIT or your choice |

**Checkpoint:**

- [ ] README renders correctly on GitHub
- [ ] All internal links work

---

### Task 5: Screenshots checklist

Capture and add to `docs/images/` (or `README.md` directly):

| Screenshot | What to show |
|------------|--------------|
| `recommender-2player.png` | Streamlit Recommender tab: 2 players, 90 min, results list |
| `chatbot-sources.png` | Rulebook Chat answer with Sources expander open |
| `mlflow-metrics.png` | MLflow UI: ALS RMSE, precision@10, hybrid params |
| `docker-pi.png` | `docker compose ps` on Raspberry Pi (or Portainer) |

**Checkpoint:**

- [ ] At least 2 screenshots in README

---

### Task 6: Release checklist

**Subtasks:**

1. Verify all step checkpoints from [docs/README.md](../README.md) progress table
2. Confirm secrets not committed:
   - `.env` gitignored
   - No `kaggle.json` in repo
   - No PDF rulebooks in git (only registry)
3. Tag release: `git tag v0.1.0` (optional)
4. Push to GitHub

**Final checklist:**

- [ ] All 7 implementation steps marked complete
- [ ] CI green
- [ ] `.env.example` documents all variables
- [ ] Docker deploy verified on Pi
- [ ] README portfolio-ready

---

## README Quick Start Snippet (for copy-paste)

```markdown
## Quick Start (Docker)

\`\`\`bash
git clone https://github.com/YOUR_USER/bgg.git && cd bgg
cp .env.example .env
# Place BGG parquet in data/processed/ and models in data/models/
# Place PDFs in rulebooks/
docker compose up -d --build
docker exec bgg-ollama ollama pull phi3:mini
docker exec bgg-api python scripts/index_rulebooks.py
open http://localhost:8501
\`\`\`

## Quick Start (Local)

\`\`\`bash
pip install -e ".[dev,spark,rag]"
python scripts/download_kaggle_data.py
python scripts/preprocess_to_parquet.py
python scripts/train_recommender.py
uvicorn bgg.api.main:app --reload &
streamlit run frontend/streamlit_app.py
\`\`\`
```

---

## Definition of Done

- [ ] GitHub Actions CI green
- [ ] README with badges, architecture, quick start, screenshots
- [ ] Release checklist complete
- [ ] Project ready for portfolio / LinkedIn

Mark Step 07 complete in [docs/README.md](../README.md).

---

## Reference

- [Implementation guide index](../README.md)
- [Folder structure](../architecture/folder-structure.md)
- [Docker Compose spec](../architecture/docker-compose-spec.md)
