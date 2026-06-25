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

Raw game data lives in `data/raw/` (gitignored). Primary file: `bga_GameItem.csv` from [mshepherd/board-games](https://www.kaggle.com/datasets/mshepherd/board-games). Re-download via `python scripts/download_kaggle_data.py` once Kaggle credentials are configured.
