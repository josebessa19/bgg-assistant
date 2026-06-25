# Step 04 — RAG Pipeline

**Week:** 3 | **Est. time:** 4–6h | **Depends on:** [01-project-setup](01-project-setup.md) | **Blocks:** [05-api-and-frontend](05-api-and-frontend.md)

Index rulebook PDFs into ChromaDB with local embeddings, implement incremental re-indexing, and wire a LangChain → Ollama Q&A chain.

---

## Objectives

- PDFs in `rulebooks/` chunked and embedded with `all-MiniLM-L6-v2`
- Persistent Chroma collection `rulebooks`
- Incremental indexer skips unchanged files
- Grounded answers from Ollama for rulebook questions

---

## Tasks

### Task 1: Game registry

**Subtasks:**

1. Create `rulebooks/game_registry.json`:

```json
[
  {
    "slug": "wingspan",
    "filename": "wingspan.pdf",
    "display_name": "Wingspan",
    "bgg_id": 266192
  }
]
```

2. Convention: PDF filename = `{slug}.pdf` unless overridden in registry
3. Add 2+ sample PDFs for development (your own rulebooks)

**Checkpoint:**

- [ ] Registry lists every PDF in `rulebooks/`
- [ ] Slugs are lowercase, no spaces

---

### Task 2: PDF chunking

**Subtasks:**

1. Create `src/bgg/rag/chunking.py`:

```python
"""Extract and chunk PDF rulebooks with PyMuPDF."""
import hashlib
import fitz  # pymupdf
from dataclasses import dataclass

CHUNK_SIZE = 500      # approximate tokens
CHUNK_OVERLAP = 50

@dataclass
class Chunk:
    chunk_id: str
    game_slug: str
    source_file: str
    page: int
    text: str
    content_hash: str

def approx_token_count(text: str) -> int:
    return len(text.split())

def extract_chunks(pdf_path: str, game_slug: str) -> list[Chunk]:
    """Page-aware sliding window chunking."""
    doc = fitz.open(pdf_path)
    chunks = []
    for page_num in range(len(doc)):
        text = doc[page_num].get_text()
        # split text into CHUNK_SIZE windows with CHUNK_OVERLAP
        ...
    return chunks

def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
```

2. Chunk ID format: `{game_slug}:{page}:{chunk_index}`
3. Skip empty pages / whitespace-only chunks

**Checkpoint:**

- [ ] Unit test: 1 PDF → chunks with correct metadata
- [ ] Average chunk size ~400–600 tokens

---

### Task 3: Embeddings and Chroma indexing

**Subtasks:**

1. Create `src/bgg/rag/rag_pipeline.py`:

```python
"""RAG pipeline: embed, index, retrieve, generate."""
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from bgg.config import settings

COLLECTION_NAME = "rulebooks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )

def get_vectorstore(persist_dir: str | None = None) -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=persist_dir or str(settings.bgg_data_dir.parent / "chroma_data"),
        client_settings=...,  # or HttpClient via CHROMA_HOST for Docker
    )

def index_chunks(chunks: list, vectorstore: Chroma) -> int:
    """Upsert chunks; return count added."""
    ...
```

2. Metadata per chunk:
   - `game_slug`, `source_file`, `page`, `chunk_id`, `content_hash`

3. Use Chroma HTTP client when `CHROMA_HOST` is set (Docker):

```python
import chromadb
from chromadb.config import Settings as ChromaSettings

def get_chroma_client():
    return chromadb.HttpClient(
        host=settings.chroma_host.replace("http://", "").split(":")[0],
        port=int(settings.chroma_host.split(":")[-1]),
    )
```

**Checkpoint:**

- [ ] Index 2+ PDFs; `collection.count() > 0`
- [ ] Query by metadata filter `game_slug=wingspan` returns only that game

---

### Task 4: Incremental indexing script

**Subtasks:**

1. Create `scripts/index_rulebooks.py`:

```python
"""Batch and incremental rulebook indexing."""
import json
import hashlib
from pathlib import Path
from bgg.config import settings
from bgg.rag.chunking import extract_chunks
from bgg.rag.rag_pipeline import get_vectorstore, delete_chunks_for_file

MANIFEST_PATH = Path("chroma_data/index_manifest.json")

def file_fingerprint(path: Path) -> dict:
    stat = path.stat()
    content = path.read_bytes()
    return {"mtime": stat.st_mtime, "sha256": hashlib.sha256(content).hexdigest()}

def load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}

def save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))

def main(incremental: bool = True) -> None:
    manifest = load_manifest()
    registry = json.loads((settings.rulebooks_dir / "game_registry.json").read_text())
    vectorstore = get_vectorstore()
    updated = 0

    for entry in registry:
        pdf_path = settings.rulebooks_dir / entry["filename"]
        if not pdf_path.exists():
            continue
        fp = file_fingerprint(pdf_path)
        key = str(pdf_path)
        if incremental and key in manifest and manifest[key] == fp:
            continue  # skip unchanged
        delete_chunks_for_file(vectorstore, entry["slug"])
        chunks = extract_chunks(str(pdf_path), entry["slug"])
        index_chunks(chunks, vectorstore)
        manifest[key] = fp
        updated += 1

  # handle removed files: keys in manifest but not on disk → delete chunks
    save_manifest(manifest)
    print(f"Updated {updated} files")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Force full reindex")
    args = parser.parse_args()
    main(incremental=not args.full)
```

2. `delete_chunks_for_file`: query Chroma by `source_file` metadata, delete IDs

**Checkpoint:**

- [ ] Re-run with no changes → `Updated 0 files`
- [ ] Add new PDF → only that file re-embedded

---

### Task 5: LangChain Q&A chain

**Subtasks:**

1. Create `src/bgg/rag/prompts.py`:

```python
RULEBOOK_SYSTEM_PROMPT = """You are a board game rules assistant.
Answer ONLY based on the provided rulebook excerpts.
If the answer is not in the context, say "I could not find that in the rulebook."
Always cite page numbers when available.
Be concise and accurate."""

RULEBOOK_USER_TEMPLATE = """Context from rulebook:
{context}

Question: {question}

Answer:"""
```

2. Add to `rag_pipeline.py`:

```python
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def build_qa_chain(game_slug: str):
  llm = Ollama(
      base_url=settings.ollama_base_url,
      model=settings.ollama_model,
      temperature=0.1,
      num_ctx=2048,
  )
  retriever = vectorstore.as_retriever(
      search_kwargs={
          "k": 4,
          "filter": {"game_slug": game_slug},
      }
  )
  ...

def query(game_slug: str, question: str, top_k: int = 4) -> dict:
    """Returns {answer, sources: [{page, chunk_id, excerpt}]}"""
    ...
```

3. Test locally with Ollama running: `ollama pull phi3:mini`

**Checkpoint:**

- [ ] Question "How does setup work?" returns grounded answer with page refs
- [ ] Off-topic question returns "could not find" message

---

### Task 6: Integration test

**Subtasks:**

1. Add `tests/test_rag_chunking.py` — chunk extraction on fixture PDF
2. Add `tests/test_rag_pipeline.py` — mock Ollama, test retrieval only (`@pytest.mark.integration` for full chain)

**Checkpoint:**

- [ ] `pytest tests/test_rag_chunking.py` passes
- [ ] Integration test passes with Ollama running (optional in CI)

---

## Ollama Model Selection

| Environment | Model | Notes |
|-------------|-------|-------|
| Raspberry Pi | `phi3:mini` | Default; fits 4–8 GB RAM |
| Dev laptop | `mistral:7b` | Better quality for prompt tuning |
| Either | `phi3:mini` | Use for parity testing before Pi deploy |

---

## Definition of Done

- [ ] 2+ PDFs indexed in Chroma
- [ ] Incremental reindex works
- [ ] Grounded Q&A via `query()` function
- [ ] Sources include page numbers

Mark Step 04 complete in [docs/README.md](../README.md).

**Next:** [05-api-and-frontend](05-api-and-frontend.md)

---

## Reference

- [Pitfalls — PDF reindex](../architecture/pitfalls-and-mitigations.md#4-pdf-re-indexing-when-adding-games)
- [Pitfalls — Ollama on Pi](../architecture/pitfalls-and-mitigations.md#1-ollama-performance-on-raspberry-pi)
- [Pitfalls — slug mismatch](../architecture/pitfalls-and-mitigations.md#6-game-name--slug--bgg-id-mismatch)
