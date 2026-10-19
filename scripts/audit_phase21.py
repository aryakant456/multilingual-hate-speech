"""Comprehensive audit script to parse, verify, and cross-check all Phase 21 metrics."""
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"

def run_audit():
    print("=" * 60)
    print("AUDIT: READING AND PARSING ALL ORIGINAL FILES")
    print("=" * 60)

    # 1. baseline_metrics.json
    with open(METRICS_DIR / "baseline_metrics.json", encoding="utf-8") as f:
        b_json = json.load(f)

    # 2. xlmr_test_metrics.json
    with open(METRICS_DIR / "xlmr_test_metrics.json", encoding="utf-8") as f:
        x_json = json.load(f)

    # 3. xlmr_external_metrics.json
    with open(METRICS_DIR / "xlmr_external_metrics.json", encoding="utf-8") as f:
        ext_json = json.load(f)

    # 4. model_comparison.csv
    mc_csv = pd.read_csv(METRICS_DIR / "model_comparison.csv")

    # 5. split_summary.csv
    split_csv = pd.read_csv(METRICS_DIR / "split_summary.csv")

    # 6. final_model_comparison.csv
    f_model_csv = pd.read_csv(METRICS_DIR / "final_model_comparison.csv")

    # 7. final_language_comparison.csv
    f_lang_csv = pd.read_csv(METRICS_DIR / "final_language_comparison.csv")

    print("\n--- 1. BASELINE METRICS AUDIT ---")
    b_test = b_json["test"]["overall"]
    b_cm = b_test["confusion_matrix"]
    print("baseline_metrics.json test overall:")
    print(f"  n = {b_test['n']}, n_hate = {b_test['n_hate']}")
    print(f"  CM = {b_cm}")
    cm_sum = sum(b_cm.values())
    print(f"  Sum of CM = {cm_sum}")
    tn_b, fp_b, fn_b, tp_b = b_cm["tn"], b_cm["fp"], b_cm["fn"], b_cm["tp"]
    print(f"  Non-hate support (tn + fp) = {tn_b + fp_b}")
    print(f"  Hate support (fn + tp) = {fn_b + tp_b}")
    print(f"  Accuracy = {b_test['accuracy']:.6f}")
    print(f"  Macro-F1 = {b_test['macro_f1']:.6f}")
    print(f"  Weighted-F1 = {b_test['weighted_f1']:.6f}")
    print(f"  f1_hate = {b_test['f1_hate']:.6f}")
    print(f"  f1_non_hate = {b_test['f1_non_hate']:.6f}")
    print(f"  precision_hate = {b_test['precision_hate']:.6f}")
    print(f"  recall_hate = {b_test['recall_hate']:.6f}")
    print(f"  ROC-AUC = {b_test.get('roc_auc')}")

    print("\n  Baseline language-level CM sum:")
    lang_sum_cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    lang_sum_n = 0
    for lg, d in b_json["test"]["by_language"].items():
        c = d["confusion_matrix"]
        print(f"    {lg}: n={d['n']}, n_hate={d['n_hate']}, CM={c}, macro-f1={d['macro_f1']:.4f}")
        for k in lang_sum_cm:
            lang_sum_cm[k] += c[k]
        lang_sum_n += d["n"]
    print(f"  Sum of language-level CMs = {lang_sum_cm} (total={sum(lang_sum_cm.values())})")
    print(f"  Sum of language n = {lang_sum_n}")

    print("\n--- 2. XLM-R TEST METRICS AUDIT ---")
    x_test = x_json["test"]["overall"]
    x_cm = x_test["confusion_matrix"]
    print("xlmr_test_metrics.json test overall:")
    print(f"  n = {x_test['n']}, n_hate = {x_test['n_hate']}")
    print(f"  CM = {x_cm}")
    print(f"  Sum of CM = {sum(x_cm.values())}")
    tn_x, fp_x, fn_x, tp_x = x_cm["tn"], x_cm["fp"], x_cm["fn"], x_cm["tp"]
    print(f"  Non-hate support (tn + fp) = {tn_x + fp_x}")
    print(f"  Hate support (fn + tp) = {fn_x + tp_x}")
    print(f"  Accuracy = {x_test['accuracy']:.6f}")
    print(f"  Macro-F1 = {x_test['macro_f1']:.6f}")
    print(f"  Weighted-F1 = {x_test['weighted_f1']:.6f}")
    print(f"  f1_hate = {x_test['f1_hate']:.6f}")
    print(f"  f1_non_hate = {x_test['f1_non_hate']:.6f}")
    print(f"  precision_hate = {x_test['precision_hate']:.6f}")
    print(f"  recall_hate = {x_test['recall_hate']:.6f}")
    print(f"  ROC-AUC = {x_test.get('roc_auc')}")

    print("\n  XLM-R language-level CM sum:")
    x_lang_sum_cm = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    for lg, d in x_json["test"]["by_language"].items():
        c = d["confusion_matrix"]
        print(f"    {lg}: n={d['n']}, n_hate={d['n_hate']}, CM={c}, macro-f1={d['macro_f1']:.4f}")
        for k in x_lang_sum_cm:
            x_lang_sum_cm[k] += c[k]
    print(f"  Sum of language-level CMs = {x_lang_sum_cm} (total={sum(x_lang_sum_cm.values())})")

    print("\n--- 3. SPLIT SUMMARY AUDIT ---")
    test_split = split_csv[split_csv["set"] == "test"]
    print(test_split)
    print(f"Total test rows in split_summary: {test_split['total'].sum()}")
    print(f"Total Hate rows in split_summary: {test_split['Hate/Abusive'].sum()}")
    print(f"Total Non-Hate rows in split_summary: {test_split['Non-Hate'].sum()}")

    print("\n--- 4. MODEL COMPARISON CSV AUDIT ---")
    print(mc_csv)

    print("\n--- 5. FINAL COMPARISON CSVS VS STORED METRICS ---")
    print("Checking final_model_comparison.csv:")
    print(f_model_csv[["Model", "Dataset", "Split", "Samples", "Accuracy", "Macro-F1", "FPR (%)", "FNR (%)"]])

    print("\nChecking final_language_comparison.csv:")
    print(f_lang_csv)

if __name__ == "__main__":
    run_audit()

