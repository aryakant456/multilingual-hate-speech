"""Prepare the 5 data files for Google Colab (Phase 8).

Run from the project root:
    python scripts/make_colab_data.py

Writes SLIM copies (only the columns the model needs) into the folder  colab_upload/ .
Upload those 5 files to Google Drive: MyDrive/hate-speech/data/ . The folder is git-ignored
(it contains real comments).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_utils import read_processed_csv, write_processed_csv  # noqa: E402
from src.utils.config import (EXTERNAL_FILE, TEST_FILE, TRAIN_FILE, VAL_FILE, ZERO_SHOT_FILE)  # noqa: E402

OUT = ROOT / "colab_upload"
PLAN = [  # (file, columns to keep)
    (TRAIN_FILE, ["text", "label", "language", "source", "split"]),
    (VAL_FILE, ["text", "label", "language", "source", "split"]),
    (TEST_FILE, ["text", "label", "language", "source", "split"]),
    (ZERO_SHOT_FILE, ["text", "label", "language", "source", "split"]),
    (EXTERNAL_FILE, ["text", "label", "language", "source", "tag", "category"]),
]


def main() -> None:
    missing = [p.name for p, _ in PLAN if not p.exists()]
    if missing:
        raise SystemExit(f"Missing: {missing}. Run phases 3-5 first (python -m src.data.split_dataset).")
    OUT.mkdir(exist_ok=True)
    total = 0.0
    for path, cols in PLAN:
        df = read_processed_csv(path)
        keep = [c for c in cols if c in df.columns]
        write_processed_csv(df[keep], OUT / path.name)
        mb = (OUT / path.name).stat().st_size / 1e6
        total += mb
        print(f"{path.name:<26}{len(df):>9,} rows  {mb:>8.1f} MB   columns: {keep}")
    print(f"\nTotal {total:.0f} MB written to {OUT}\nNext: upload these 5 files to Google Drive -> MyDrive/hate-speech/data/")


if __name__ == "__main__":
    main()
