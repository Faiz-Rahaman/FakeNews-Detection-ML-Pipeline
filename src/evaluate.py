import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from ensemble import StackingEnsemble, WeightedEnsemble
from feature_engineering import HybridFeatureBuilder
from preprocessing import load_true_fake_dataset


LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def find_best_f1_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, prob)
    f1 = (2 * precision * recall) / np.maximum(precision + recall, 1e-12)
    if len(thresholds) == 0:
        return 0.5, float(f1[0])
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx]), float(f1[:-1][best_idx])


def classification_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    best_t, best_f1 = find_best_f1_threshold(y_true, prob)
    unique_classes = np.unique(y_true)
    has_both_classes = len(unique_classes) >= 2

    roc_auc = float(roc_auc_score(y_true, prob)) if has_both_classes else 0.5
    pr_auc = float(average_precision_score(y_true, prob)) if has_both_classes else 0.0
    logloss = float(log_loss(y_true, prob, labels=[0, 1]))

    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "log_loss": logloss,
        "brier": float(brier_score_loss(y_true, prob)),
        "best_f1_threshold": best_t,
        "best_f1": best_f1,
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            pred,
            labels=[0, 1],
            target_names=["REAL", "FAKE"],
            output_dict=True,
            digits=4,
            zero_division=0,
        ),
    }


def plot_calibration_curves(
    y_true: np.ndarray,
    probs: dict[str, np.ndarray],
    out_path: Path,
    n_bins: int = 10,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label=name)

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curves")
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_reliability_diagram(
    y_true: np.ndarray,
    probs: dict[str, np.ndarray],
    out_path: Path,
    n_bins: int = 12,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1.2, label="Perfect")

    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
        axes[0].plot(mean_pred, frac_pos, marker="o", linewidth=2, label=name)
        axes[1].hist(p, bins=n_bins, alpha=0.35, label=name)

    axes[0].set_ylabel("Observed frequency")
    axes[0].set_title("Reliability Diagram")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best")
    axes[1].set_xlabel("Predicted probability (FAKE)")
    axes[1].set_ylabel("Count")
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_json(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_ablation_table(rows: list[dict[str, Any]], out_csv: Path, out_md: Path) -> None:
    df = pd.DataFrame(rows)
    if "f1" in df.columns:
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    md_lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    for _, row in df.iterrows():
        md_lines.append("| " + " | ".join(str(row[c]) for c in df.columns) + " |")

    out_md.write_text("\n".join(md_lines), encoding="utf-8")


def build_model_comparison_table(metrics_by_model: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for name, m in metrics_by_model.items():
        report = m["classification_report"]
        rows.append(
            {
                "model": name,
                "f1_overall": m["f1"],
                "f1_real": report["REAL"]["f1-score"],
                "f1_fake": report["FAKE"]["f1-score"],
                "roc_auc": m["roc_auc"],
                "pr_auc": m["pr_auc"],
                "tn": m["confusion_matrix"]["tn"],
                "fp": m["confusion_matrix"]["fp"],
                "fn": m["confusion_matrix"]["fn"],
                "tp": m["confusion_matrix"]["tp"],
            }
        )
    return pd.DataFrame(rows).sort_values("f1_overall", ascending=False).reset_index(drop=True)


def predict_bert_prob(model_dir: Path, texts: np.ndarray, batch_size: int = 16) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = list(texts[i : i + batch_size])
            enc = tokenizer(batch, truncation=True, padding=True, max_length=384, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            batch_prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            probs.append(batch_prob)

    return np.concatenate(probs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate research fake-news model artifact")
    parser.add_argument("--artifact_path", default="models/research/final_model.joblib")
    parser.add_argument("--true_csv", default="True.csv")
    parser.add_argument("--fake_csv", default="Fake.csv")
    parser.add_argument("--output_dir", default="models/research/eval")
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    setup_logging("INFO")
    args = parse_args()

    artifact = joblib.load(args.artifact_path)
    threshold = float(args.threshold) if args.threshold is not None else float(artifact.get("threshold", 0.5))

    df = load_true_fake_dataset(Path(args.true_csv), Path(args.fake_csv))
    X_text = df["combined_text"].values
    y = df["label"].astype(int).values

    feature_builder = HybridFeatureBuilder.from_artifacts(artifact["feature_artifacts"])
    X = feature_builder.transform(df["combined_text"])

    prob_linear = artifact["linear_svc"].predict_proba(X)[:, 1]
    prob_xgb = artifact["xgboost"].predict_proba(X)[:, 1] if "xgboost" in artifact else np.full_like(prob_linear, 0.5)

    bert_dir = Path(artifact.get("bert_model_dir", ""))
    if bert_dir.exists():
        prob_bert = predict_bert_prob(bert_dir, X_text)
    else:
        LOGGER.warning("BERT directory not found; using 0.5 fallback probabilities")
        prob_bert = np.full_like(prob_linear, 0.5)

    if "stacking_meta_model" in artifact:
        stacker = StackingEnsemble()
        stacker.meta_model = artifact["stacking_meta_model"]
        stacker.constant_prob = artifact.get("stacking_constant_prob", None)
        stacker.use_average = bool(artifact.get("stacking_use_average", False))
        base = np.column_stack([prob_linear, prob_bert])
        prob_ensemble = stacker.predict_proba(base)
    else:
        w = artifact["ensemble_weights"]
        ensemble = WeightedEnsemble(w["linear"], w["xgboost"], w["bert"])
        prob_ensemble = ensemble.predict_proba(prob_linear, prob_xgb, prob_bert)

    report = {
        "linear_svc": classification_metrics(y, prob_linear, threshold),
        "xgboost": classification_metrics(y, prob_xgb, threshold),
        "distilbert": classification_metrics(y, prob_bert, threshold),
        "ensemble": classification_metrics(y, prob_ensemble, threshold),
    }

    out_dir = Path(args.output_dir)
    save_json(report, out_dir / "evaluation_report.json")
    plot_calibration_curves(
        y_true=y,
        probs={
            "LinearSVC": prob_linear,
            "DistilBERT": prob_bert,
            "Ensemble": prob_ensemble,
        },
        out_path=out_dir / "calibration_curve.png",
    )
    plot_reliability_diagram(
        y_true=y,
        probs={
            "LinearSVC": prob_linear,
            "DistilBERT": prob_bert,
            "Ensemble": prob_ensemble,
        },
        out_path=out_dir / "reliability_diagram.png",
    )

    comparison = build_model_comparison_table(
        {
            "linear_svc": report["linear_svc"],
            "distilbert": report["distilbert"],
            "ensemble": report["ensemble"],
        }
    )
    comparison.to_csv(out_dir / "model_comparison.csv", index=False)

    print(json.dumps({k: {"f1": v["f1"], "roc_auc": v["roc_auc"]} for k, v in report.items()}, indent=2))
    print(f"Saved evaluation report to: {out_dir / 'evaluation_report.json'}")


if __name__ == "__main__":
    main()
