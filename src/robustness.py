import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import f1_score


@dataclass
class RobustnessReport:
    flip_rate: float
    mean_probability_shift: float
    f1_against_original_labels: float


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def slight_wording_variation(text: str) -> str:
    replacements = {
        "said": "stated",
        "announced": "declared",
        "report": "article",
        "reports": "articles",
        "fake": "false",
        "real": "genuine",
        "government": "administration",
        "official": "authority",
    }

    out = text
    for src, dst in replacements.items():
        out = re.sub(rf"\b{re.escape(src)}\b", dst, out, flags=re.IGNORECASE)
    out = out.replace(".", ";") if "." in out else out
    return _normalize_whitespace(out)


def paraphrase_style(text: str) -> str:
    prefix = "In summary, "
    out = text
    out = re.sub(r"\bhowever\b", "but", out, flags=re.IGNORECASE)
    out = re.sub(r"\btherefore\b", "so", out, flags=re.IGNORECASE)
    out = re.sub(r"\bin addition\b", "also", out, flags=re.IGNORECASE)
    return _normalize_whitespace(prefix + out)


def debunk_style(text: str) -> str:
    sentence = text.split(".")[0][:280]
    return _normalize_whitespace(
        f"Fact check: The claim that {sentence} is false. Evidence indicates this claim is misleading and not true."
    )


def evaluate_variant(
    base_prob: np.ndarray,
    variant_prob: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
) -> RobustnessReport:
    base_pred = (base_prob >= threshold).astype(int)
    variant_pred = (variant_prob >= threshold).astype(int)
    flip_rate = float(np.mean(base_pred != variant_pred))
    mean_shift = float(np.mean(np.abs(base_prob - variant_prob)))
    f1 = float(f1_score(y_true, variant_pred, zero_division=0))
    return RobustnessReport(flip_rate=flip_rate, mean_probability_shift=mean_shift, f1_against_original_labels=f1)


def run_robustness_suite(
    texts: list[str],
    y_true: np.ndarray,
    predict_proba_fn: Callable[[list[str]], np.ndarray],
    threshold: float,
    sample_size: int = 1200,
    random_state: int = 42,
) -> dict:
    if len(texts) == 0:
        return {"error": "No texts provided"}

    rng = random.Random(random_state)
    indices = list(range(len(texts)))
    if len(indices) > sample_size:
        indices = rng.sample(indices, sample_size)

    sampled_texts = [texts[i] for i in indices]
    sampled_y = y_true[indices]

    base_prob = predict_proba_fn(sampled_texts)

    slight_prob = predict_proba_fn([slight_wording_variation(t) for t in sampled_texts])
    paraphrase_prob = predict_proba_fn([paraphrase_style(t) for t in sampled_texts])
    debunk_prob = predict_proba_fn([debunk_style(t) for t in sampled_texts])

    slight = evaluate_variant(base_prob, slight_prob, sampled_y, threshold)
    paraphrase = evaluate_variant(base_prob, paraphrase_prob, sampled_y, threshold)

    debunk_real_rate = float(np.mean((debunk_prob < threshold).astype(int)))
    debunk_mean_fake_prob = float(np.mean(debunk_prob))

    return {
        "sample_size": len(sampled_texts),
        "threshold": threshold,
        "slight_wording_variation": slight.__dict__,
        "paraphrase": paraphrase.__dict__,
        "debunk_style": {
            "mean_fake_probability": debunk_mean_fake_prob,
            "predicted_real_rate": debunk_real_rate,
        },
    }


def save_robustness_report(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
