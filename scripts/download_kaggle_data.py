"""Download optional BGG datasets from Kaggle into data/raw/.

Primary raw files for this project (placed manually in data/raw/):
  - games_detailed_info2025.csv — game metadata (~28k rows, BGG API-style columns)
  - bgg-26m-reviews.csv         — user ratings + reviews (~26M rows)

Optional Kaggle fallback: mshepherd/board-games
https://www.kaggle.com/datasets/mshepherd/board-games

Credentials: ~/.kaggle/access_token or ~/.kaggle/kaggle.json (not stored in this repo)

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
