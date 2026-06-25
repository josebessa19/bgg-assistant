"""Download BGG dataset from Kaggle into data/raw/.

Dataset: mshepherd/board-games (Recommend.Games / board-game-scraper subset)
https://www.kaggle.com/datasets/mshepherd/board-games

Credentials: ~/.kaggle/access_token or ~/.kaggle/kaggle.json (not stored in this repo)

Files in data/raw/ (as of 2026-06):
  - bga_GameItem.csv  — game metadata (~124k rows): bgg_id, name, players, time,
    category, mechanic, cooperative, bayes_rating, complexity, etc.

Optional from same Kaggle dataset (needed for ALS in Step 03):
  - bgg_RatingItem.csv — user–game ratings (large; download when training recommender)
  - bgg_GameItem.csv   — BGG-only game table (alternative to bga_GameItem)

Manual download: place CSVs in data/raw/ if the CLI is unavailable.
"""
import subprocess
import sys
from pathlib import Path

DATASET = "mshepherd/board-games"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            DATASET,
            "-p",
            str(RAW_DIR),
            "--unzip",
        ],
        check=True,
    )
    print(f"Downloaded to {RAW_DIR}")


if __name__ == "__main__":
    main()
