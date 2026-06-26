# BGG Board Game Assistant

Self-hosted board game recommender and rulebook RAG chatbot powered by BGG data, PySpark MLlib, ChromaDB, and Ollama.

**Implementation guides:** see [docs/README.md](docs/README.md) for step-by-step build instructions.

## Dev setup

```powershell
cd bgg
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,spark,rag]"
copy .env.example .env
```

Raw game data lives in `data/raw/` (gitignored):

- `games_detailed_info2025.csv` — game metadata (~28k rows)
- `bgg-26m-reviews.csv` — user ratings (~26M rows)

Preprocess to Parquet: `python scripts/preprocess_to_parquet.py`

## Spark on Windows (optional but recommended)

Jupyter runs with the notebook folder as cwd; paths are resolved from the **project root** via `bgg.config` (restart kernel after pulling updates).

| Install | Why |
|---------|-----|
| **[Eclipse Temurin JDK 17 (64-bit)](https://adoptium.net/)** | Replaces 32-bit Java; allows `spark.driver.memory` > 1g and faster 26M-row EDA |
| Set `JAVA_HOME` | Point to the 64-bit JDK, e.g. `C:\Program Files\Eclipse Adoptium\jdk-17...` |

After installing 64-bit Java, verify: `java -version` should show **64-Bit Server VM**, not Client VM.

The `NativeCodeLoader` warning is harmless on Windows. `tools/hadoop/bin/` (winutils) is already wired in for local Parquet I/O.
