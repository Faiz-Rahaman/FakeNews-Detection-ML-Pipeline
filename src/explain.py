import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from feature_engineering import HybridFeatureBuilder, feature_names_suffix
from preprocessing import load_true_fake_dataset


LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _classical_feature_names(builder: HybridFeatureBuilder) -> list[str]:
    word_names = list(builder.word_vectorizer.get_feature_names_out())
    char_names = [f"char::{x}" for x in builder.char_vectorizer.get_feature_names_out()]
    return word_names + char_names + feature_names_suffix()


def explain_linear(linear_model, X_sample, feature_names, out_dir: Path):
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)

    base_estimator = None
    if hasattr(linear_model, "calibrated_classifiers_") and linear_model.calibrated_classifiers_:
        base_estimator = linear_model.calibrated_classifiers_[0].estimator

    if base_estimator is None or not hasattr(base_estimator, "coef_"):
        raise ValueError("Could not access linear model coefficients for SHAP")

    explainer = shap.LinearExplainer(base_estimator, X_sample, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(out_dir / "linear_shap_summary.png", dpi=180)
    plt.close()

    abs_mean = np.asarray(np.abs(shap_values).mean(axis=0)).ravel()
    top_idx = np.argsort(abs_mean)[-25:][::-1]
    top = [{"feature": feature_names[i], "mean_abs_shap": float(abs_mean[i])} for i in top_idx]
    (out_dir / "linear_top_features.json").write_text(json.dumps(top, indent=2), encoding="utf-8")


def explain_xgboost(xgb_model, X_sample, feature_names, out_dir: Path):
    import shap

    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False, max_display=25)
    plt.tight_layout()
    plt.savefig(out_dir / "xgb_shap_summary.png", dpi=180)
    plt.close()

    abs_mean = np.asarray(np.abs(shap_values).mean(axis=0)).ravel()
    top_idx = np.argsort(abs_mean)[-25:][::-1]
    top = [{"feature": feature_names[i], "mean_abs_shap": float(abs_mean[i])} for i in top_idx]
    (out_dir / "xgb_top_features.json").write_text(json.dumps(top, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP explainability for classical fake-news models")
    parser.add_argument("--artifact_path", default="models/research/final_model.joblib")
    parser.add_argument("--true_csv", default="True.csv")
    parser.add_argument("--fake_csv", default="Fake.csv")
    parser.add_argument("--sample_size", type=int, default=1200)
    parser.add_argument("--output_dir", default="models/research/explain")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main():
    setup_logging("INFO")
    args = parse_args()

    artifact = joblib.load(args.artifact_path)
    feature_builder = HybridFeatureBuilder.from_artifacts(artifact["feature_artifacts"])

    df = load_true_fake_dataset(Path(args.true_csv), Path(args.fake_csv))
    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=args.random_state).reset_index(drop=True)

    X = feature_builder.transform(df["combined_text"])
    X_shap = X.tocsr() if sparse.issparse(X) else np.asarray(X)

    feature_names = _classical_feature_names(feature_builder)
    out_dir = Path(args.output_dir)

    LOGGER.info("Generating SHAP for LinearSVC")
    explain_linear(artifact["linear_svc"], X_shap, feature_names, out_dir)

    if "xgboost" in artifact:
        LOGGER.info("Generating SHAP for XGBoost")
        explain_xgboost(artifact["xgboost"], X_shap, feature_names, out_dir)
    else:
        LOGGER.info("No XGBoost model found in artifact; skipping tree SHAP output.")

    print(f"Saved SHAP outputs to: {out_dir}")


if __name__ == "__main__":
    main()
