# Research-Grade Fake News Detection

This project is now focused on real-world robustness instead of only benchmark accuracy.

## What is implemented

- Leakage reduction:
  - strips location headers, agency formatting, publisher patterns, and journalist bylines
- Split strategy improvements:
  - time-based split (default) for temporal realism
  - optional source-based split
  - cross-domain holdout validation on unseen sources
- Semantic understanding:
  - DistilBERT fine-tuning (`--run_bert`)
- Hybrid ensembling:
  - calibrated TF-IDF + LinearSVC probability
  - DistilBERT probability
  - stacking meta-classifier (Logistic Regression)
- Claim-detection features (in dense engineered block):
  - number count
  - named-entity proxy count
  - source cue count
  - citation marker count
- Reliability analysis:
  - calibration curve
  - reliability diagram
- Robustness stress testing:
  - paraphrased inputs
  - slight wording variation
  - debunk-style article formulation
- Metrics outputs:
  - confusion matrix
  - per-class F1
  - model comparison table for LinearSVC vs DistilBERT vs Ensemble

## Project modules (`src/`)

- `preprocessing.py`: leakage cleanup + time/source/cross-domain splitting
- `feature_engineering.py`: stylometric + readability + sentiment + claim-detection features
- `ensemble.py`: stacking/ensemble logic
- `train.py`: end-to-end training + cross-domain + robustness + artifact save
- `evaluate.py`: artifact evaluation + calibration/reliability + comparison table
- `robustness.py`: perturbation-based stress tests
- `predict.py`: confidence-aware inference
- `explain.py`: SHAP for classical models

## Setup

```bash
pip install -r requirements.txt
```

## Train (recommended robust run)

```bash
python src/train.py \
  --true_csv True.csv \
  --fake_csv Fake.csv \
  --output_dir models/research \
  --split_method time \
  --run_bert \
  --bert_use_gpu \
  --bert_epochs 1 \
  --bert_max_length 192 \
  --word_max_features 150000 \
  --char_max_features 120000 \
  --run_cross_domain \
  --run_ablation
```

For quick iteration (still keeps BERT enabled), sample data:

```bash
python src/train.py \
  --true_csv True.csv \
  --fake_csv Fake.csv \
  --output_dir models/research_fast \
  --split_method time \
  --run_bert \
  --bert_use_gpu \
  --bert_epochs 1 \
  --bert_max_length 128 \
  --word_max_features 80000 \
  --char_max_features 60000 \
  --sample_size 12000
```

Key outputs in `models/research/`:
- `final_model.joblib`
- `metrics_report.json`
- `model_comparison.csv`
- `calibration_curve.png`
- `reliability_diagram.png`
- `robustness_report.json`
- `cross_domain_report.json` (when `--run_cross_domain`)
- `cross_domain_model_comparison.csv` (when `--run_cross_domain`)
- `ablation.csv`, `ablation.md` (when `--run_ablation`)
- `distilbert/` (when `--run_bert`)

## Evaluate saved artifact

```bash
python src/evaluate.py \
  --artifact_path models/research/final_model.joblib \
  --true_csv True.csv \
  --fake_csv Fake.csv \
  --output_dir models/research/eval
```

## Predict with confidence

```bash
python src/predict.py \
  --model_path models/research/final_model.joblib \
  --text "Paste article text here"
```

## SHAP explainability

```bash
python src/explain.py \
  --artifact_path models/research/final_model.joblib \
  --true_csv True.csv \
  --fake_csv Fake.csv \
  --output_dir models/research/explain
```

## Reproducibility

- fixed random seed via `--random_state`
- deterministic split config saved in reports
- exported JSON/CSV artifacts for auditability
