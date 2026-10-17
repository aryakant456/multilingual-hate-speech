# NLP-Based Framework for Hate/Abusive Speech Detection in Multilingual Social Media (XLM-R Transfer Learning)

Binary hate/abusive speech detection (Non-Hate vs Hate/Abusive) for Hindi, Tamil, Malayalam and English
using XLM-RoBERTa fine-tuning, with explainability (Integrated Gradients) and a FastAPI backend.

## Datasets (not stored in this repo)
| Dataset | Source |
|---|---|
| MACD (Hindi, Tamil, Malayalam, Telugu, Kannada) | https://github.com/ShareChatAI/MACD |
| HateXplain (English) | https://github.com/hate-alert/HateXplain |
| IndicGuard (Indic LLM-safety corpus) | https://huggingface.co/datasets/l3cube-pune/IndicGuard |

Datasets are git-ignored (size, licences, abusive content). Download with the script below.

## Setup
```bash
python -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS / Linux:       source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name hate-speech --display-name "Python (hate-speech)"
```

## Phase 2: get and inspect the data
```bash
python scripts/download_data.py all      # or: macd | hatexplain | indicguard
python scripts/inspect_raw.py            # prints real columns / labels of every raw file
jupyter lab                              # then run notebooks/01_dataset_analysis.ipynb
```

## Dataset notes (verified on the real files)
- **MACD**: only `data/raw/macd/dataset/` is used (the other folders are re-splits of the same comments).
  Columns `text`, `label`; **label 0 = abusive, 1 = non-abusive** (matches the paper's counts: 74,550 / 77,872).
  Our unified `label_binary` flips this so 1 = Hate/Abusive. Languages: Hindi, Tamil, Malayalam, Telugu, Kannada.
- **HateXplain**: `dataset.json` + `post_id_divisions.json`; majority vote of 3 annotators; normal -> 0, offensive/hatespeech -> 1.
- **IndicGuard**: `<Language>/{train,valid,test}.json`; text = `prompt`; hate = `violated_categories` contains `Hate/Identity Hate`;
  repeated ids and empty prompts are dropped. It is an LLM-safety corpus (machine-translated), best used as an external test set.

## Layout
See the folder tree: `src/` (code), `notebooks/` (analysis), `backend/` (API), `results/` (metrics, figures).

## Status
- [x] Phase 1 - environment & structure
- [x] Phase 2 - dataset acquisition & inspection
- [ ] Phase 3+ - preprocessing, baseline, XLM-R training, evaluation, explainability, API

## Ethics note
The data contains offensive language and is used here solely for building moderation tools.


## Pipeline (Phases 3-7)
Run from the project root with the virtual environment active:
```
python scripts/check_phase_1_2.py              # are phases 1-2 complete?
python -m src.data.combine_datasets            # Phase 3: unified labels      -> data/processed/unified_all.csv
python -m src.data.preprocess                  # Phase 4: cleaning + dedupe   -> data/processed/clean_all.csv
python -m src.data.split_dataset               # Phase 5: train/val/test/zero-shot/external
python -m src.baseline.train_baseline          # Phase 7: TF-IDF + Logistic Regression baseline
```
Notebooks: `02_preprocessing.ipynb` (EDA, Phase 6) and `03_baseline.ipynb` (Phase 7).
Documentation: `docs/label_mapping.md`, `docs/preprocessing.md`.
Data design: train/val/test = MACD Hindi+Tamil+Malayalam + HateXplain English (official splits);
zero-shot = MACD Telugu+Kannada; external = IndicGuard (never touched before final evaluation).
