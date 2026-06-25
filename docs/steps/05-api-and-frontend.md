# Step 05 — API & Frontend

**Week:** 3–4 | **Est. time:** 5–8h | **Depends on:** [03-recommender](03-recommender.md), [04-rag-pipeline](04-rag-pipeline.md) | **Blocks:** [06-docker-and-deploy](06-docker-and-deploy.md)

Expose recommender and RAG via FastAPI endpoints and build a Streamlit frontend with two tabs.

---

## Objectives

- FastAPI app with `/health`, `/recommend`, `/ask`
- Pydantic request/response models with validation
- Streamlit UI: Recommender tab + Rulebook Chat tab
- Contract tests for API

---

## API Contracts

### `GET /health`

```json
{"status": "ok", "recommender_loaded": true, "chroma_connected": true}
```

### `POST /recommend`

**Request:**

```json
{
  "liked_game_ids": [174430, 161936],
  "players": 2,
  "max_duration_minutes": 90,
  "mode": "any",
  "top_k": 10
}
```

| Field | Type | Constraints |
|-------|------|-------------|
| `liked_game_ids` | `list[int]` | min 1, max 50 |
| `players` | `int` | 1–20 |
| `max_duration_minutes` | `int` | 15–600 |
| `mode` | `str` | `cooperative`, `competitive`, `any` |
| `top_k` | `int` | 1–50, default 10 |

**Response:**

```json
{
  "recommendations": [
    {
      "game_id": 123456,
      "name": "Example Game",
      "score": 0.87,
      "reason": "Similar mechanics to liked games; fits 2 players",
      "als_score": 0.82,
      "content_score": 0.95,
      "playing_time": 60,
      "bayes_average": 7.4
    }
  ],
  "cold_start": false
}
```

**Errors:**

- `422` — invalid IDs listed in `detail`
- `503` — model not loaded

### `POST /ask`

**Request:**

```json
{
  "game_slug": "wingspan",
  "question": "How does setup work?",
  "top_k_chunks": 4
}
```

| Field | Type | Constraints |
|-------|------|-------------|
| `game_slug` | `str` | must exist in `game_registry.json` |
| `question` | `str` | min 3, max 500 chars |
| `top_k_chunks` | `int` | 1–8, default 4 |

**Response:**

```json
{
  "answer": "Setup involves placing the bird tray...",
  "sources": [
    {
      "page": 3,
      "chunk_id": "wingspan:3:0",
      "excerpt": "Place the bird tray in the center..."
    }
  ],
  "game_slug": "wingspan"
}
```

**Errors:**

- `404` — slug not in registry or no indexed chunks
- `503` — Ollama unreachable

### `GET /games/search?q={query}` (optional helper for Streamlit)

```json
{
  "results": [
    {"game_id": 266192, "name": "Wingspan"}
  ]
}
```

---

## Tasks

### Task 1: FastAPI application shell

**Subtasks:**

1. Create `src/bgg/api/main.py`:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from bgg.api.recommender_api import router as recommender_router
from bgg.api.chatbot_api import router as chatbot_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    # load recommender artifacts, verify chroma connection
    app.state.recommender_ready = True
    yield

app = FastAPI(title="BGG Board Game Assistant", lifespan=lifespan)
app.include_router(recommender_router, tags=["recommender"])
app.include_router(chatbot_router, tags=["chatbot"])

@app.get("/health")
def health():
    return {"status": "ok"}
```

2. Run locally: `uvicorn bgg.api.main:app --reload --port 8000`

**Checkpoint:**

- [ ] `curl http://localhost:8000/health` returns 200
- [ ] OpenAPI docs at `/docs`

---

### Task 2: Recommender endpoint

**Subtasks:**

1. Create `src/bgg/api/recommender_api.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from bgg.recommender.inference import hybrid_recommend, validate_game_ids

router = APIRouter()

class RecommendRequest(BaseModel):
    liked_game_ids: list[int] = Field(..., min_length=1, max_length=50)
    players: int = Field(2, ge=1, le=20)
    max_duration_minutes: int = Field(90, ge=15, le=600)
    mode: str = Field("any", pattern="^(cooperative|competitive|any)$")
    top_k: int = Field(10, ge=1, le=50)

class RecommendationItem(BaseModel):
    game_id: int
    name: str
    score: float
    reason: str
    als_score: float | None = None
    content_score: float | None = None
    playing_time: int | None = None
    bayes_average: float | None = None

class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]
    cold_start: bool

@router.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    invalid = validate_game_ids(req.liked_game_ids)
    if invalid:
        raise HTTPException(422, detail={"invalid_game_ids": invalid})
    results = hybrid_recommend(
        liked_ids=req.liked_game_ids,
        players=req.players,
        max_duration=req.max_duration_minutes,
        mode=req.mode,
        top_k=req.top_k,
    )
    cold_start = len(req.liked_game_ids) < 3
    return RecommendResponse(
        recommendations=[RecommendationItem(**r) for r in results],
        cold_start=cold_start,
    )
```

2. Load game lookup parquet at module init for ID validation

**Checkpoint:**

- [ ] `curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"liked_game_ids":[174430,161936],"players":2,"max_duration_minutes":90,"mode":"any","top_k":5}'` returns JSON with 5 games

---

### Task 3: Chatbot endpoint

**Subtasks:**

1. Create `src/bgg/api/chatbot_api.py`:

```python
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from bgg.rag.rag_pipeline import query
from bgg.rag.registry import slug_exists  # helper to check game_registry.json

router = APIRouter()

class AskRequest(BaseModel):
    game_slug: str = Field(..., min_length=1)
    question: str = Field(..., min_length=3, max_length=500)
    top_k_chunks: int = Field(4, ge=1, le=8)

class SourceItem(BaseModel):
    page: int
    chunk_id: str
    excerpt: str

class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    game_slug: str

@router.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not slug_exists(req.game_slug):
        raise HTTPException(404, detail=f"Unknown game slug: {req.game_slug}")
    try:
        result = query(req.game_slug, req.question, top_k=req.top_k_chunks)
    except ConnectionError:
        raise HTTPException(503, detail="Ollama service unavailable")
    if not result["sources"]:
        raise HTTPException(404, detail="No indexed content for this game")
    return AskResponse(**result)
```

**Checkpoint:**

- [ ] `curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"game_slug":"wingspan","question":"How does setup work?"}'` returns answer + sources

---

### Task 4: Streamlit frontend

**Subtasks:**

1. Create `frontend/streamlit_app.py`:

```python
import os
import httpx
import streamlit as st

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")

st.set_page_config(page_title="BGG Assistant", layout="wide")
st.title("BGG Board Game Assistant")

tab_rec, tab_chat = st.tabs(["Recommender", "Rulebook Chat"])

with tab_rec:
    st.subheader("Discover your next game")
    # Load game names for multiselect (from parquet or /games/search)
    liked = st.multiselect("Games you enjoy", options=game_names, ...)
    col1, col2, col3 = st.columns(3)
    players = col1.number_input("Players", 1, 10, 2)
    duration = col2.number_input("Max duration (min)", 15, 300, 90)
    mode = col3.selectbox("Mode", ["any", "cooperative", "competitive"])
    if st.button("Get recommendations"):
        resp = httpx.post(f"{API_URL}/recommend", json={...}, timeout=30)
        for rec in resp.json()["recommendations"]:
            st.write(f"**{rec['name']}** — score {rec['score']:.2f}")
            st.caption(rec["reason"])

with tab_chat:
    st.subheader("Ask about the rules")
    slug = st.selectbox("Game", options=slugs_from_registry)
    question = st.text_input("Your question")
    if st.button("Ask") and question:
        resp = httpx.post(f"{API_URL}/ask", json={"game_slug": slug, "question": question})
        data = resp.json()
        st.write(data["answer"])
        with st.expander("Sources"):
            for s in data["sources"]:
                st.write(f"Page {s['page']}: {s['excerpt'][:200]}...")
```

2. Cache game list with `@st.cache_data`
3. Show cold-start hint when &lt; 3 games selected

**Checkpoint:**

- [ ] `streamlit run frontend/streamlit_app.py` works against local API
- [ ] Both tabs functional

---

### Task 5: API tests

**Subtasks:**

1. Create `tests/test_api.py`:

```python
from fastapi.testclient import TestClient
from bgg.api.main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").status_code == 200

def test_recommend_invalid_id():
    resp = client.post("/recommend", json={
        "liked_game_ids": [999999999],
        "players": 2,
        "max_duration_minutes": 90,
        "mode": "any",
    })
    assert resp.status_code == 422

def test_ask_unknown_slug():
    resp = client.post("/ask", json={"game_slug": "nonexistent", "question": "setup?"})
    assert resp.status_code == 404
```

2. Mock `hybrid_recommend` and `query` in unit tests; integration tests optional

**Checkpoint:**

- [ ] `pytest tests/test_api.py` passes

---

## Local Dev Workflow

```bash
# Terminal 1 — API
uvicorn bgg.api.main:app --reload --port 8000

# Terminal 2 — Streamlit
FASTAPI_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py

# Terminal 3 — Ollama (for chat)
ollama serve
ollama pull phi3:mini
```

---

## Definition of Done

- [ ] `/recommend` returns valid JSON with scores and reasons
- [ ] `/ask` returns answer + page sources
- [ ] Streamlit both tabs call API successfully
- [ ] API tests pass

Mark Step 05 complete in [docs/README.md](../README.md).

**Next:** [06-docker-and-deploy](06-docker-and-deploy.md)

---

## Reference

- [API contracts above](#api-contracts)
- [Pitfalls — invalid BGG IDs](../architecture/pitfalls-and-mitigations.md#7-invalid-bgg-ids-in-recommend-request)
