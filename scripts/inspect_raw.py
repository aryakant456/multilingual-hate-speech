"""Print the real structure (files, columns, dtypes, nulls, label values) of each
raw dataset so we never have to guess column names.

Usage:  python scripts/inspect_raw.py
Writes a copy of the report to results/metrics/raw_structure_report.txt
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.io_utils import find_data_files, inspect_frame, read_any  # noqa: E402
from src.utils.config import (HATEXPLAIN_DIR, INDICGUARD_DIR, MACD_DIR,  # noqa: E402
                              METRICS_DIR)


def report(name: str, root: Path) -> None:
    print("=" * 80, f"\n{name}  ({root})\n" + "=" * 80)
    files = find_data_files(root)
    if not files:
        print("  (no data files found - download first)\n")
        return
    for f in files:
        try:
            df = read_any(f)
        except Exception as exc:  # noqa: BLE001
            print(f"--- {f.relative_to(root)}: unreadable ({exc})\n")
            continue
        inspect_frame(df, str(f.relative_to(root)))
        for col in df.columns:
            try:
                n = df[col].nunique()
            except TypeError:
                continue
            if 1 < n <= 15:
                print(f"  value counts for '{col}':")
                print(df[col].value_counts(dropna=False).to_string())
        print()


if __name__ == "__main__":
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report("MACD", MACD_DIR)
        report("HateXplain", HATEXPLAIN_DIR)
        report("IndicGuard", INDICGUARD_DIR)
    text = buf.getvalue()
    print(text)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "raw_structure_report.txt").write_text(text, encoding="utf-8")
    print(f"Report saved to {METRICS_DIR / 'raw_structure_report.txt'}")
