"""Check that Phases 1 and 2 are really complete.

Run from the project root (with the (.venv) active):
    python scripts/check_phase_1_2.py            # quick (does not load the big IndicGuard files)
    python scripts/check_phase_1_2.py --full     # also loads IndicGuard (takes a few minutes)

Prints PASS / WARN / FAIL for every check. FAIL = fix before continuing.
WARN = unusual (e.g. a number differs from what we saw before) but not necessarily wrong.
"""
from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
results: list[tuple[str, str, str]] = []


def add(status: str, name: str, detail: str = "") -> None:
    results.append((status, name, detail))
    print(f"[{status:<4}] {name}" + (f"  -  {detail}" if detail else ""))


def main(full: bool) -> None:
    # ---- Phase 1 ---------------------------------------------------------------
    add("PASS" if sys.version_info >= (3, 10) else "FAIL", "Python version", sys.version.split()[0])
    for pkg in ["numpy", "pandas", "pyarrow", "matplotlib", "seaborn", "sklearn", "huggingface_hub", "ipykernel", "tqdm"]:
        try:
            importlib.import_module(pkg)
            add("PASS", f"package {pkg}")
        except ImportError:
            add("FAIL", f"package {pkg}", "run: pip install -r requirements.txt")

    folders = ["data/raw/macd", "data/raw/hatexplain", "data/raw/indicguard", "data/processed", "data/external_test",
               "notebooks", "src/data", "src/baseline", "src/model", "src/explainability", "src/utils", "models",
               "results/figures", "results/metrics", "results/predictions", "backend", "frontend"]
    missing = [f for f in folders if not (ROOT / f).is_dir()]
    add("PASS" if not missing else "FAIL", "project folders", f"missing: {missing}" if missing else f"{len(folders)} folders present")
    nbs = [f"notebooks/0{i}_{n}.ipynb" for i, n in enumerate(
        ["dataset_analysis", "preprocessing", "baseline", "xlm_training", "evaluation", "explainability"], 1)]
    miss_nb = [n for n in nbs if not (ROOT / n).exists()]
    add("PASS" if not miss_nb else "FAIL", "notebooks 01-06", f"missing: {miss_nb}" if miss_nb else "all 6 present")
    for f in ["requirements.txt", "README.md", ".gitignore"]:
        add("PASS" if (ROOT / f).exists() else "FAIL", f)

    # ---- Phase 2: raw files -------------------------------------------------------
    macd = sorted((ROOT / "data/raw/macd/dataset").glob("*.csv"))
    add("PASS" if len(macd) == 15 else "FAIL", "MACD files in data/raw/macd/dataset/",
        f"{len(macd)} csv files (expected 15 = 5 languages x train/val/test)")
    hx = [(ROOT / "data/raw/hatexplain" / n).exists() for n in ("dataset.json", "post_id_divisions.json")]
    add("PASS" if all(hx) else "FAIL", "HateXplain files", "dataset.json + post_id_divisions.json" if all(hx) else "missing")
    ig_dir = ROOT / "data/raw/indicguard"
    ig_langs = sorted(p.name for p in ig_dir.iterdir() if p.is_dir() and not p.name.startswith(".")
                      and (p / "train.json").exists()) if ig_dir.exists() else []
    add("PASS" if len(ig_langs) == 11 else "WARN", "IndicGuard language folders", f"{len(ig_langs)} found (expected 11): {ig_langs}")

    # ---- Phase 2: loaders give the numbers we verified ---------------------------------
    try:
        from src.data.load_hatexplain import load_hatexplain
        from src.data.load_macd import load_macd
        m = load_macd(verbose=False)
        n_ab = int((m["label_raw"].astype(int) == 0).sum())
        add("PASS" if (len(m), n_ab) == (152422, 74550) else "WARN", "MACD rows / abusive rows",
            f"{len(m):,} / {n_ab:,} (we saw 152,422 / 74,550 = the paper's numbers)")
        add("PASS" if set(m["language"]) == {"Hindi", "Tamil", "Malayalam", "Telugu", "Kannada"} else "FAIL",
            "MACD languages recognised", str(sorted(set(m["language"]))))
        add("PASS" if set(m["split"]) == {"train", "val", "test"} else "FAIL", "MACD official splits detected", str(sorted(set(m["split"]))))
        h = load_hatexplain(verbose=False)
        add("PASS" if len(h) == 20148 else "WARN", "HateXplain posts", f"{len(h):,} (we saw 20,148)")
        # HateXplain's official split file only lists posts that have a majority label. The posts where all
        # 3 annotators disagree ("undecided") are not in it -> split "unknown". They are dropped in Phase 3,
        # so that is fine - but ONLY if every "unknown" post really is undecided. We verify that here.
        unk = h[h["split"] == "unknown"]
        real = set(h["split"]) - {"unknown"}
        all_undecided = bool((unk["label_raw"] == "undecided").all())
        if real == {"train", "val", "test"} and all_undecided:
            add("PASS", "HateXplain official splits detected",
                f"{len(h) - len(unk):,} posts have an official split; {len(unk):,} posts have none because "
                "all 3 annotators disagreed ('undecided') - they are dropped in Phase 3")
        else:
            bad = unk[unk["label_raw"] != "undecided"]
            add("FAIL", "HateXplain official splits detected",
                f"{len(bad):,} labelled posts have no official split (splits found: {sorted(real)})")
    except Exception as exc:  # noqa: BLE001
        add("FAIL", "loading MACD / HateXplain", f"{type(exc).__name__}: {exc}")

    if full:
        try:
            from src.data.load_indicguard import load_indicguard
            g = load_indicguard(verbose=False)
            add("PASS" if len(g) == 431464 else "WARN", "IndicGuard rows after cleaning", f"{len(g):,} (we saw 431,464)")
            add("PASS" if g["language"].nunique() == 11 else "FAIL", "IndicGuard languages", str(g["language"].nunique()))
        except Exception as exc:  # noqa: BLE001
            add("FAIL", "loading IndicGuard", f"{type(exc).__name__}: {exc}")
    else:
        add("WARN", "IndicGuard loader not run", "use --full to load it too (slow)")

    # ---- notebook 01 outputs --------------------------------------------------------------
    n_met = len(list((ROOT / "results/metrics").glob("*.csv")))
    n_fig = len(list((ROOT / "results/figures").glob("*.png")))
    add("PASS" if n_met and n_fig else "WARN", "notebook 01 outputs", f"{n_met} csv in results/metrics, {n_fig} png in results/figures")

    # ---- git ---------------------------------------------------------------------------------
    try:
        if not (ROOT / ".git").exists():
            add("WARN", "git repository", "not initialised yet (run: git init) - needed before pushing")
        else:
            def ignored(p: str) -> bool:
                return subprocess.run(["git", "check-ignore", "-q", p], cwd=ROOT).returncode == 0
            add("PASS" if ignored("data/raw/macd/dataset/hindi_train.csv") else "FAIL", "raw data ignored by git")
            rep = "results/metrics/raw_structure_report.txt"
            tracked = subprocess.run(["git", "ls-files", rep], cwd=ROOT, capture_output=True, text=True).stdout.strip()
            if tracked:
                add("FAIL", "raw_structure_report.txt is tracked by git",
                    f"it contains real abusive text. Run: git rm --cached {rep}")
            else:
                add("PASS" if ignored(rep) else "WARN", "raw_structure_report.txt ignored by git",
                    "" if ignored(rep) else "it contains real abusive text - add this line to .gitignore: " + rep)
    except FileNotFoundError:
        add("WARN", "git", "git command not found")

    c = {s: sum(1 for r in results if r[0] == s) for s in ("PASS", "WARN", "FAIL")}
    print(f"\nSUMMARY: {c['PASS']} PASS, {c['WARN']} WARN, {c['FAIL']} FAIL")
    print("=> Phases 1-2 are COMPLETE." if c["FAIL"] == 0 else "=> Fix the FAIL lines above, then run this again.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    main(ap.parse_args().full)
