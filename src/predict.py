import argparse
from pathlib import Path

import joblib
import numpy as np

from ensemble import StackingEnsemble, WeightedEnsemble
from feature_engineering import HybridFeatureBuilder


def bert_probabilities(model_dir: Path, texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    enc = tokenizer(texts, truncation=True, padding=True, max_length=384, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    return torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()


def load_legacy_or_research(path: Path):
    artifact = joblib.load(path)
    if isinstance(artifact, dict) and str(artifact.get("version", "")).startswith(("2.", "3.")):
        return artifact, "research"
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact, "legacy"
    return {"model": artifact, "threshold": 0.5, "label_map": {0: "REAL", 1: "FAKE"}}, "legacy"


def predict_research(artifact: dict, text: str) -> tuple[float, str]:
    feature_builder = HybridFeatureBuilder.from_artifacts(artifact["feature_artifacts"])
    X = feature_builder.transform([text])

    p_linear = artifact["linear_svc"].predict_proba(X)[:, 1]

    bert_dir = Path(artifact.get("bert_model_dir", ""))
    if bert_dir.exists():
        p_bert = bert_probabilities(bert_dir, [text])
    else:
        p_bert = np.array([0.5], dtype=np.float64)

    if "stacking_meta_model" in artifact:
        stack = StackingEnsemble()
        stack.meta_model = artifact["stacking_meta_model"]
        stack.constant_prob = artifact.get("stacking_constant_prob", None)
        stack.use_average = bool(artifact.get("stacking_use_average", False))
        p = float(stack.predict_proba(np.column_stack([p_linear, p_bert]))[0])
    else:
        p_xgb = artifact["xgboost"].predict_proba(X)[:, 1]
        w = artifact["ensemble_weights"]
        ensemble = WeightedEnsemble(w["linear"], w["xgboost"], w["bert"])
        p = float(ensemble.predict_proba(p_linear, p_xgb, p_bert)[0])

    threshold = float(artifact.get("threshold", 0.5))
    pred = int(p >= threshold)
    label = artifact.get("label_map", {0: "REAL", 1: "FAKE"}).get(pred, "FAKE" if pred == 1 else "REAL")
    return p, label


def main():
    parser = argparse.ArgumentParser(description="Predict fake news probability")
    parser.add_argument("--model_path", default="models/research/final_model.joblib")
    parser.add_argument("--text", required=True)
    args = parser.parse_args()

    artifact, mode = load_legacy_or_research(Path(args.model_path))

    if mode == "research":
        p_fake, label = predict_research(artifact, args.text)
        confidence = max(p_fake, 1.0 - p_fake)
        print(f"Mode: research-ensemble")
        print(f"P(fake): {p_fake:.4f}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Prediction: {label}")
        return

    model = artifact["model"]
    threshold = float(artifact.get("threshold", 0.5))
    p_fake = float(model.predict_proba([args.text])[0][1])
    pred = int(p_fake >= threshold)
    label = artifact.get("label_map", {0: "REAL", 1: "FAKE"}).get(pred, "FAKE" if pred == 1 else "REAL")
    confidence = max(p_fake, 1.0 - p_fake)

    print("Mode: legacy")
    print(f"P(fake): {p_fake:.4f}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Prediction: {label}")


if __name__ == "__main__":
    main()
