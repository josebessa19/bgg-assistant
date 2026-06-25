# Pitfalls and Mitigations

Known risks for this stack on a home server (Raspberry Pi) and how to avoid them.

## 1. Ollama Performance on Raspberry Pi

**Risk:** Mistral 7B exceeds RAM on 4–8 GB Pi; inference takes 30–120s per answer; OOM kills container.

**Mitigations:**

| Action | Detail |
|--------|--------|
| Use `phi3:mini` | Default on Pi; ~2.3 GB quantized |
| Set memory limit | `deploy.resources.limits.memory: 3g` on ollama service |
| Single concurrent request | `OLLAMA_NUM_PARALLEL=1`; no streaming until stable |
| Reduce context | `num_ctx=2048` in LangChain Ollama config |
| Pre-warm on boot | `docker exec bgg-ollama ollama run phi3:mini "ok"` in cron or compose entrypoint |
| Dev vs prod model | Train/index on laptop with `mistral:7b`; Pi runs `phi3:mini` only |
| Remote Ollama | Point `OLLAMA_BASE_URL` to a desktop on LAN if Pi is too slow |

**Checkpoint signal:** Answer latency &lt; 60s for a 4-chunk retrieval on Pi with `phi3:mini`.

---

## 2. PySpark on Raspberry Pi

**Risk:** JVM + Spark driver consumes 1–2 GB RAM; ALS training on 20k+ games is impractical on Pi.

**Mitigations:**

| Action | Detail |
|--------|--------|
| Train offline | Run `02_recommender.ipynb` and `train_recommender.py` on laptop |
| Export artifacts | Save factor matrices + content index to `data/models/` |
| API inference only | `inference.py` loads numpy/pickle artifacts — no SparkSession in API container |
| Optional `implicit` lib | Lightweight ALS inference alternative if Spark export is heavy |

**Checkpoint signal:** API container RSS &lt; 512 MB without Spark installed.

---

## 3. ALS Cold Start (Few User Ratings)

**Risk:** User provides 1–2 liked games; ALS user vector is unreliable or undefined.

**Mitigations:**

| Action | Detail |
|--------|--------|
| Threshold | If `len(liked_game_ids) < 3`, skip ALS branch |
| Content fallback | Mean feature vector of liked games → cosine similarity |
| Popularity tie-break | Weight by `bayes_average` within filter constraints |
| UI hint | Streamlit: "Add 3+ games for better collaborative filtering" |
| Hybrid when ready | `score = 0.6 * als + 0.4 * content` only when ALS is active |

**Checkpoint signal:** 1 liked game + `players=2` still returns 10 filtered, plausible games.

---

## 4. PDF Re-indexing When Adding Games

**Risk:** Full re-embed of all PDFs on every deploy is slow (CPU-bound MiniLM).

**Mitigations:**

| Action | Detail |
|--------|--------|
| Content-hash manifest | `chroma_data/index_manifest.json`: `{path: {mtime, sha256}}` |
| Skip unchanged | If hash matches, skip file entirely |
| Incremental upsert | New/changed PDFs only; delete Chroma IDs for removed files |
| Stable chunk IDs | `{game_slug}:{page}:{chunk_index}` for idempotent upsert |
| Admin trigger | Optional `POST /admin/reindex` or cron `index_rulebooks.py --incremental` |
| Naming convention | `rulebooks/{game_slug}.pdf` + `game_registry.json` |

**Checkpoint signal:** Re-run indexer with no changes → log shows `0 files updated, 0 chunks embedded`.

---

## 5. ChromaDB + LangChain Version Drift

**Risk:** Breaking API changes between `chromadb`, `langchain`, and `langchain-community`.

**Mitigations:**

| Action | Detail |
|--------|--------|
| Pin versions | Lock in `pyproject.toml`; regenerate lockfile on upgrade |
| Integration test | `tests/test_rag_chunking.py` with one fixture PDF |
| HTTP client mode | Use `CHROMA_HOST` HTTP client in API; same as compose service |
| Collection schema | Document metadata fields; migration script if schema changes |

---

## 6. Game Name ↔ Slug ↔ BGG ID Mismatch

**Risk:** User selects "Wingspan" in UI but PDF is `wingspan.pdf`; BGG ID lookup fails.

**Mitigations:**

| Action | Detail |
|--------|--------|
| `game_registry.json` | `{slug, filename, bgg_id, display_name}` |
| Streamlit search | Load game names from `games_features.parquet`; map to `game_id` |
| Chat tab | Dropdown of slugs from registry (not free text) |
| Validation | API returns 404 with clear message if slug has no indexed PDF |

---

## 7. Invalid BGG IDs in Recommend Request

**Risk:** User typos or outdated IDs break inference.

**Mitigations:**

| Action | Detail |
|--------|--------|
| Validate at API | Check IDs exist in parquet index before scoring |
| Return 422 | List invalid IDs in error response |
| Fuzzy match (UI) | Streamlit `st.multiselect` with search over game names |

---

## 8. Large Rulebook PDFs

**Risk:** 100+ page PDFs produce thousands of chunks; slow indexing and noisy retrieval.

**Mitigations:**

| Action | Detail |
|--------|--------|
| Chunk size | ~500 tokens, overlap 50 |
| Metadata filter | Always filter Chroma by `game_slug` before similarity search |
| `top_k_chunks` | Default 4; cap at 8 in API |
| Page-aware citations | Include `page` in response `sources[]` |

---

## 9. MLflow Artifact Path in Docker

**Risk:** Model loaded from `file://` path that differs between host and container.

**Mitigations:**

| Action | Detail |
|--------|--------|
| MLflow server in compose | API uses `http://mlflow:5000` |
| Register model | `models:/bgg-recommender/Production` URI |
| Fallback | Copy artifacts to `data/models/` for offline Pi deploy |

---

## 10. Memory Pressure (All Services on Pi)

**Risk:** Ollama + Chroma + API + MLflow + Streamlit exceed total RAM.

**Mitigations:**

| Action | Detail |
|--------|--------|
| Run MLflow on laptop | Pi API loads from `data/models/` only; omit mlflow service on Pi |
| Swap | 2 GB swap file |
| Stagger services | Don't hit chat + recommend simultaneously during demo |
| Streamlit on laptop | Pi runs api + ollama + chroma; browse Streamlit from dev machine |

**Minimal Pi stack:** `ollama` + `chromadb` + `api` (drop `mlflow` and `streamlit` from Pi compose override).

---

## Quick Reference Table

| Risk | Severity on Pi | Primary mitigation |
|------|----------------|-------------------|
| Ollama OOM | High | `phi3:mini`, memory limit |
| Spark on Pi | High | Train on laptop |
| ALS cold start | Medium | Content + popularity fallback |
| PDF reindex | Medium | Hash manifest, incremental |
| Version drift | Low | Pin deps, integration tests |
| Slug mismatch | Medium | `game_registry.json` |
| Invalid BGG IDs | Low | API validation |
| Large PDFs | Medium | Metadata filter by game |
| MLflow paths | Medium | Model registry URI |
| Total RAM | High | Minimal Pi stack option |
