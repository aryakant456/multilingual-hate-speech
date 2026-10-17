"""Download the three raw datasets into data/raw/.

Usage (run from the project root, venv activated):
    python scripts/download_data.py hatexplain
    python scripts/download_data.py macd
    python scripts/download_data.py indicguard
    python scripts/download_data.py all

If a download fails (network, gated dataset, changed URL) the script prints
manual instructions - you can always place the files in data/raw/<name>/ by hand.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import HATEXPLAIN_DIR, INDICGUARD_DIR, MACD_DIR  # noqa: E402

HATEXPLAIN_BASE = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"
HATEXPLAIN_FILES = ["dataset.json", "post_id_divisions.json"]
MACD_REPO = "https://github.com/ShareChatAI/MACD.git"
INDICGUARD_HF_REPO = "l3cube-pune/IndicGuard"


def download_hatexplain() -> None:
    HATEXPLAIN_DIR.mkdir(parents=True, exist_ok=True)
    for name in HATEXPLAIN_FILES:
        dest = HATEXPLAIN_DIR / name
        print(f"[HateXplain] downloading {name} ...")
        urllib.request.urlretrieve(HATEXPLAIN_BASE + name, dest)
        print(f"  saved -> {dest}  ({dest.stat().st_size/1e6:.1f} MB)")


def download_macd() -> None:
    MACD_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"[MACD] cloning {MACD_REPO} ...")
        subprocess.run(["git", "clone", "--depth", "1", MACD_REPO, tmp + "/macd"], check=True)
        src = Path(tmp) / "macd"
        shutil.rmtree(src / ".git", ignore_errors=True)  # no nested git repo
        for item in src.iterdir():
            target = MACD_DIR / item.name
            if target.exists():
                shutil.rmtree(target) if target.is_dir() else target.unlink()
            shutil.move(str(item), target)
    print(f"  copied repository contents -> {MACD_DIR}")
    print("  NOTE: open the README inside that folder to see how the data is organised;")
    print("  if data files look like tiny text pointers (Git LFS), download them manually")
    print("  from https://github.com/ShareChatAI/MACD instead.")


def download_indicguard() -> None:
    from huggingface_hub import snapshot_download

    INDICGUARD_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[IndicGuard] downloading {INDICGUARD_HF_REPO} from Hugging Face ...")
    snapshot_download(repo_id=INDICGUARD_HF_REPO, repo_type="dataset",
                      local_dir=str(INDICGUARD_DIR))
    print(f"  saved -> {INDICGUARD_DIR}")
    print("  If access is denied, run `huggingface-cli login` (and accept the dataset's terms")
    print("  on its Hugging Face page if it is gated), then re-run.")


JOBS = {"hatexplain": download_hatexplain, "macd": download_macd,
        "indicguard": download_indicguard}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=[*JOBS, "all"])
    args = ap.parse_args()
    for name in (JOBS if args.dataset == "all" else [args.dataset]):
        try:
            JOBS[name]()
        except Exception as exc:  # noqa: BLE001
            print(f"[{name}] FAILED: {exc}\n  -> download it manually into data/raw/{name}/")


if __name__ == "__main__":
    main()
