# Step 06 — Docker & Deploy

**Week:** 4 | **Est. time:** 4–6h | **Depends on:** [05-api-and-frontend](05-api-and-frontend.md) | **Blocks:** [07-ci-cd-and-readme](07-ci-cd-and-readme.md)

Containerize all services, deploy to home server (Raspberry Pi) with Docker Compose, and verify end-to-end.

---

## Objectives

- `docker-compose.yml` with ollama, chromadb, mlflow, api, streamlit
- Dockerfiles for API and Streamlit
- Persistent volumes for models, chroma, rulebooks, mlflow
- Stack runs on dev machine and Raspberry Pi

---

## Tasks

### Task 1: API Dockerfile

**Subtasks:**

1. Create `docker/Dockerfile.api`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir -e ".[rag]"

COPY scripts/ scripts/
COPY frontend/ frontend/

# Pre-download embedding model (optional, speeds first index)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000
CMD ["uvicorn", "bgg.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

2. Note: Spark **not** installed in API image — inference uses `data/models/` artifacts only

**Checkpoint:**

- [ ] `docker build -f docker/Dockerfile.api -t bgg-api .` succeeds

---

### Task 2: Streamlit Dockerfile

**Subtasks:**

1. Create `docker/Dockerfile.streamlit`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY frontend/ frontend/
RUN pip install --no-cache-dir -e . streamlit httpx

EXPOSE 8501
CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

**Checkpoint:**

- [ ] `docker build -f docker/Dockerfile.streamlit -t bgg-streamlit .` succeeds

---

### Task 3: Docker Compose

**Subtasks:**

1. Copy full spec from [docker-compose-spec.md](../architecture/docker-compose-spec.md) to project root `docker-compose.yml`
2. Ensure volume mounts:
   - `./data` → `/data` (ro) — must contain `processed/` and `models/` before API start
   - `./rulebooks` → `/rulebooks` (ro)
   - `./chroma_data` → chroma persistence
   - `./mlruns` → mlflow
   - `ollama_models` named volume
3. Add `healthcheck` on API service
4. Create `docker-compose.pi.yml` override (optional):

```yaml
# docker-compose.pi.yml — use: docker compose -f docker-compose.yml -f docker-compose.pi.yml up -d
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 3g
  mlflow:
    profiles: ["full"]  # omit on Pi: docker compose --profile full ...
```

**Checkpoint:**

- [ ] `docker compose config` validates without errors

---

### Task 4: Environment file for Docker

**Subtasks:**

1. Ensure `.env` exists (from `.env.example`)
2. For Docker, compose `environment:` block overrides service URLs — verify no `localhost` leaks inside containers
3. Document required pre-deploy data:
   - `data/processed/games_features.parquet`
   - `data/models/*` (exported from Step 03)
   - `rulebooks/*.pdf` + `game_registry.json`

**Checkpoint:**

- [ ] `.env.example` matches all compose env vars

---

### Task 5: Bootstrap sequence

**Subtasks:**

1. First-time deploy script or documented commands:

```bash
# Build and start
docker compose up -d --build

# Pull LLM
docker exec bgg-ollama ollama pull phi3:mini

# Index rulebooks
docker exec bgg-api python scripts/index_rulebooks.py

# Verify
curl http://localhost:8000/health
curl http://localhost:8501
```

2. Pre-warm Ollama (optional cron on Pi):

```bash
docker exec bgg-ollama ollama run phi3:mini "ready"
```

**Checkpoint:**

- [ ] All containers `running` (`docker compose ps`)
- [ ] Health endpoint returns `recommender_loaded: true`

---

### Task 6: Raspberry Pi deployment

**Subtasks:**

1. Copy to Pi via git clone or rsync:
   - Repo code
   - `data/processed/` and `data/models/` (from laptop training)
   - `rulebooks/` PDFs
2. Pi prerequisites:
   - Docker + Docker Compose v2
   - 2 GB swap enabled
   - Ports 8000, 8501, 11434 available (or remapped)
3. Use `phi3:mini` only on Pi
4. **Minimal Pi stack** (if RAM constrained): drop `mlflow` service; API loads from `data/models/` directly

| Full stack | Minimal Pi |
|------------|------------|
| ollama, chroma, mlflow, api, streamlit | ollama, chroma, api |
| MLflow UI on :5000 | Artifacts baked into `data/models/` |
| Streamlit on Pi | Streamlit on laptop → `FASTAPI_URL=http://pi:8000` |

5. Test from LAN: `http://<pi-ip>:8501`

**Checkpoint:**

- [ ] Streamlit reachable on LAN
- [ ] Recommend + ask work from Pi deployment
- [ ] Answer latency &lt; 60s with `phi3:mini`

---

### Task 7: Operational notes

**Subtasks:**

1. Document in this file or a `docs/operations.md` (optional):
   - Restart: `docker compose restart api`
   - Logs: `docker compose logs -f api ollama`
   - Reindex after new PDF: `docker exec bgg-api python scripts/index_rulebooks.py`
   - Update model: rsync new `data/models/`, restart api

**Checkpoint:**

- [ ] Restart api container does not lose Chroma data (volume persists)

---

## Pi Resource Budget (estimate)

| Service | RAM |
|---------|-----|
| Ollama (phi3:mini) | ~2.5 GB |
| Chroma | ~200 MB |
| API | ~400 MB |
| Streamlit | ~150 MB |
| MLflow | ~200 MB |
| **Total** | ~3.5 GB + OS |

On 4 GB Pi: use minimal stack without MLflow/Streamlit on device.

---

## Definition of Done

- [ ] `docker compose up -d` healthy on dev machine
- [ ] Same compose works on Pi (full or minimal)
- [ ] Streamlit usable via browser
- [ ] Volumes persist across restarts

Mark Step 06 complete in [docs/README.md](../README.md).

**Next:** [07-ci-cd-and-readme](07-ci-cd-and-readme.md)

---

## Reference

- [Docker Compose spec](../architecture/docker-compose-spec.md)
- [Pitfalls — Ollama on Pi](../architecture/pitfalls-and-mitigations.md#1-ollama-performance-on-raspberry-pi)
- [Pitfalls — memory pressure](../architecture/pitfalls-and-mitigations.md#10-memory-pressure-all-services-on-pi)
