"""Central configuration: paths, label maps and column overrides.

After you inspect the raw data (Phase 2), fill in the *_COL overrides below
if the automatic column detection picks the wrong column.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MACD_DIR = RAW_DIR / "macd"
HATEXPLAIN_DIR = RAW_DIR / "hatexplain"
INDICGUARD_DIR = RAW_DIR / "indicguard"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_TEST_DIR = DATA_DIR / "external_test"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"

RANDOM_SEED = 42

# ------------------------------------------------- unified table schema
UNIFIED_COLUMNS = ["text", "label_raw", "label_binary", "language", "source", "split"]
LABEL_NAMES = {0: "Non-Hate", 1: "Hate/Abusive"}

# ------------------------------------- automatic column detection lists
# Checked in order, case-insensitive. First match wins.
TEXT_COL_CANDIDATES = ["text", "comment", "comment_text", "commenttext", "content",
                       "sentence", "tweet", "post", "message", "prompt", "response"]
LABEL_COL_CANDIDATES = ["label", "labels", "class", "target", "abusive",
                        "is_abusive", "hate", "toxic", "safety_label", "prompt_label"]
LANG_COL_CANDIDATES = ["language", "lang", "lang_code", "language_code"]
CATEGORY_COL_CANDIDATES = ["category", "categories", "domain", "harm_category",
                           "violated_categories", "subcategory", "topic"]

# ------------------------------------------------ manual overrides
# Set to a column name (string) to force it. None = auto-detect.
MACD_TEXT_COL = None
MACD_LABEL_COL = None

# ---------------------------------------------------------------- MACD
# Verified against the real files (Phase 2 report) and the NeurIPS-2022 paper:
#   * data/raw/macd/ holds 5 variants of the SAME 152K comments:
#       dataset/ (official balanced train/val/test)  <-- we use this one
#       dataset_meta/ (same + user_id, post_id), dataset_80_10_10/, dataset_chrono/,
#       dataset_unbalanced/   (reading all of them would count each comment ~5x)
#   * Columns: `text`, `label`.  IMPORTANT: label 0 = ABUSIVE, label 1 = NON-abusive.
#     (sum of label==0 over dataset/ is 74,550 = the paper's "abusive samples";
#      label==1 is 77,872 = the paper's "non-abusive samples")
MACD_SUBDIR = "dataset"
MACD_ABUSIVE_LABEL = 0
INDICGUARD_TEXT_COL = None
INDICGUARD_LABEL_COL = None
INDICGUARD_CATEGORY_COL = None
INDICGUARD_LANG_COL = None

# --------------------------------------------------------- languages
LANGUAGE_ALIASES = {
    "hindi": "Hindi", "hi": "Hindi",
    "tamil": "Tamil", "ta": "Tamil",
    "malayalam": "Malayalam", "ml": "Malayalam",
    "telugu": "Telugu", "te": "Telugu",
    "kannada": "Kannada", "kn": "Kannada",
    "bengali": "Bengali", "bn": "Bengali",
    "gujarati": "Gujarati", "gu": "Gujarati",
    "marathi": "Marathi", "mr": "Marathi",
    "punjabi": "Punjabi", "pa": "Punjabi",
    "urdu": "Urdu", "ur": "Urdu",
    "english": "English", "en": "English",
    # spellings that really occur in the downloaded files (folder/file names + data)
    "malyalam": "Malayalam", "gujrati": "Gujarati", "punajbi": "Punjabi",
}

# ------------------------------------------------ HateXplain mapping
# HateXplain labels: hatespeech / offensive / normal.
# Deliberate decision: our task is binary "Hate/Abusive" vs "Non-Hate",
# so hatespeech AND offensive -> 1, normal -> 0. The original 3-class label
# is always kept in the `label_raw` column so you can change this later.
HATEXPLAIN_LABEL_MAP = {"normal": 0, "offensive": 1, "hatespeech": 1}

# ------------------------------------------------ IndicGuard mapping
# IndicGuard is an LLM *safety* dataset (AEGIS 2.0 taxonomy). The column
# `violated_categories` holds a COMMA-SEPARATED list per row, e.g.
#     "Hate/Identity Hate, Harassment"
# A row counts as Hate/Abusive (label 1) if ANY of its categories is in the set
# below (lower-case). Rows whose prompt is "safe" -> 0. Rows that are unsafe for
# OTHER reasons (e.g. only "Criminal Planning/Confessions") stay NaN and are NOT
# used as non-hate, because they are not "normal" text either.
# Verify the real category names in notebook 01, section 9, then adjust.
# Broader option: add "harassment", "profanity", "threat".
INDICGUARD_HATE_CATEGORIES = {"hate/identity hate"}


# ================================================================ Phases 3-7
# ---- which data is used for what (Phase 5 design decision) ----------------
# Train / validation / test : MACD Hindi + Tamil + Malayalam  and  HateXplain English
#                             (OFFICIAL splits are kept, nothing is re-shuffled)
# Zero-shot test            : MACD Telugu + Kannada  (never seen in training)
# External test             : IndicGuard (never touched until final evaluation)
TRAIN_MACD_LANGUAGES = ["Hindi", "Tamil", "Malayalam"]
ZERO_SHOT_LANGUAGES = ["Telugu", "Kannada"]
SPLIT_RANK = {"train": 0, "val": 1, "test": 2}          # priority when removing duplicates
MACD_PAPER_COUNTS = {"abusive": 74550, "non_abusive": 77872}   # MACD paper (NeurIPS 2022)

# ---- files written by the pipeline ----------------------------------------
UNIFIED_FILE = PROCESSED_DIR / "unified_all.csv"                 # Phase 3 (MACD + HateXplain)
INDICGUARD_UNIFIED_FILE = EXTERNAL_TEST_DIR / "indicguard_unified.csv"
CLEAN_FILE = PROCESSED_DIR / "clean_all.csv"                     # Phase 4
INDICGUARD_CLEAN_FILE = EXTERNAL_TEST_DIR / "indicguard_clean.csv"
TRAIN_FILE = PROCESSED_DIR / "train.csv"                         # Phase 5
VAL_FILE = PROCESSED_DIR / "val.csv"
TEST_FILE = PROCESSED_DIR / "test.csv"
ZERO_SHOT_FILE = PROCESSED_DIR / "zero_shot_test.csv"
EXTERNAL_FILE = EXTERNAL_TEST_DIR / "indicguard_external.csv"    # final external test set
BASELINE_DIR = MODELS_DIR / "baseline"


# ================================================================ Phases 8-13 (XLM-R)
XLMR_MODEL_NAME = "xlm-roberta-base"
XLMR_DIR = MODELS_DIR / "xlm_roberta_hate"        # the fine-tuned model lives here (git-ignored)
ID2LABEL = {0: "NON-HATE", 1: "HATE"}
LABEL2ID = {"NON-HATE": 0, "HATE": 1}
INTERNAL_LANGUAGES = ["English", "Hindi", "Tamil", "Malayalam"]   # language-wise internal evaluation
EXTERNAL_LANGUAGES = ["English", "Hindi", "Tamil", "Malayalam"]   # IndicGuard languages we evaluate
