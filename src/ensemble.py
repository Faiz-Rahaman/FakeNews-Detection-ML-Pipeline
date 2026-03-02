from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass
class WeightedEnsemble:
    linear_weight: float
    xgb_weight: float
    bert_weight: float

    def __post_init__(self) -> None:
        s = self.linear_weight + self.xgb_weight + self.bert_weight
        if s <= 0:
            raise ValueError("Sum of ensemble weights must be > 0")
        self.linear_weight /= s
        self.xgb_weight /= s
        self.bert_weight /= s

    def predict_proba(self, linear_prob: np.ndarray, xgb_prob: np.ndarray, bert_prob: np.ndarray) -> np.ndarray:
        linear_prob = np.asarray(linear_prob, dtype=np.float64)
        xgb_prob = np.asarray(xgb_prob, dtype=np.float64)
        bert_prob = np.asarray(bert_prob, dtype=np.float64)
        return (
            self.linear_weight * linear_prob
            + self.xgb_weight * xgb_prob
            + self.bert_weight * bert_prob
        )

    @classmethod
    def from_validation_auc(cls, auc_linear: float, auc_xgb: float, auc_bert: float) -> "WeightedEnsemble":
        scores = np.array([auc_linear, auc_xgb, auc_bert], dtype=np.float64)
        scores = np.clip(scores, 1e-6, None)
        # Sharpening gives stronger influence to better-performing models.
        scores = scores ** 2
        return cls(scores[0], scores[1], scores[2])


class StackingEnsemble:
    """Logistic stacking over model probabilities."""

    def __init__(self, random_state: int = 42) -> None:
        self.constant_prob: float | None = None
        self.use_average: bool = False
        self.meta_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
        )

    def fit(self, base_probs: np.ndarray, y: np.ndarray) -> "StackingEnsemble":
        y_arr = np.asarray(y)
        unique = np.unique(y_arr)
        if len(unique) < 2:
            # Degenerate validation fold: fallback to average of base probabilities.
            self.constant_prob = None
            self.use_average = True
            return self
        self.constant_prob = None
        self.use_average = False
        self.meta_model.fit(base_probs, y_arr)
        return self

    def predict_proba(self, base_probs: np.ndarray) -> np.ndarray:
        if self.use_average:
            base = np.asarray(base_probs, dtype=np.float64)
            return np.mean(base, axis=1)
        if self.constant_prob is not None:
            return np.full(shape=(len(base_probs),), fill_value=self.constant_prob, dtype=np.float64)
        return self.meta_model.predict_proba(base_probs)[:, 1]
