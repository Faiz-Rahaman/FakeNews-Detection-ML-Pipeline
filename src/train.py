import argparse
import logging
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from ensemble import StackingEnsemble
from evaluate import (
    build_model_comparison_table,
    classification_metrics,
    plot_calibration_curves,
    plot_reliability_diagram,
    save_ablation_table,
    save_json,
)
from feature_engineering import HybridFeatureBuilder
from preprocessing import (
    SplitConfig,
    load_true_fake_dataset,
    set_seed,
    source_holdout_split,
    split_dataset,
)
from robustness import run_robustness_suite, save_robustness_report


LOGGER = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def train_linear_svc(X_train, y_train, random_state: int) -> CalibratedClassifierCV:
    base = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=random_state,
        max_iter=20000,
        tol=1e-4,
    )
    model = CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
    model.fit(X_train, y_train)
    return model


def train_distilbert(
    train_texts,
    train_labels,
    val_texts,
    val_labels,
    model_dir: Path,
    random_state: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    max_length: int,
    use_gpu: bool,
):
    import torch
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = Dataset.from_dict({"text": list(train_texts), "label": list(train_labels)})
    val_ds = Dataset.from_dict({"text": list(val_texts), "label": list(val_labels)})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]
        preds = (probs >= 0.5).astype(int)
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            roc_auc = 0.5
        else:
            roc_auc = float(roc_auc_score(labels, probs))
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision_score(labels, preds, zero_division=0)),
            "recall": float(recall_score(labels, preds, zero_division=0)),
            "f1": float(f1_score(labels, preds, zero_division=0)),
            "roc_auc": roc_auc,
        }

    has_cuda = torch.cuda.is_available()
    effective_use_gpu = use_gpu and has_cuda
    if use_gpu and not has_cuda:
        LOGGER.warning("GPU requested for BERT but CUDA is not available. Falling back to CPU.")
    LOGGER.info("DistilBERT device=%s", "cuda" if effective_use_gpu else "cpu")

    args = TrainingArguments(
        output_dir=str(model_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=random_state,
        logging_steps=50,
        report_to=[],
        no_cuda=not effective_use_gpu,
        fp16=effective_use_gpu,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    return trainer


def distilbert_predict_proba(trainer, texts, max_length: int) -> np.ndarray:
    from datasets import Dataset
    import torch

    dataset = Dataset.from_dict({"text": list(texts), "label": [0] * len(texts)})

    def tokenize(batch):
        return trainer.tokenizer(batch["text"], truncation=True, max_length=max_length)

    dataset = dataset.map(tokenize, batched=True)
    pred = trainer.predict(dataset)
    return torch.softmax(torch.tensor(pred.predictions), dim=1).numpy()[:, 1]


def run_ablation(train_texts, train_y, val_texts, val_y, out_dir: Path, random_state: int) -> None:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from scipy import sparse

    rows = []

    word = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        stop_words="english",
        sublinear_tf=True,
        max_features=220000,
    )
    X_tr_word = word.fit_transform(train_texts)
    X_va_word = word.transform(val_texts)
    m1 = LinearSVC(C=1.0, class_weight="balanced", random_state=random_state).fit(X_tr_word, train_y)
    rows.append({"split": "val", "model": "word_only_linear", "f1": float(f1_score(val_y, m1.predict(X_va_word)))})

    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, max_features=220000)
    X_tr_char = char.fit_transform(train_texts)
    X_va_char = char.transform(val_texts)
    X_tr_wc = sparse.hstack([X_tr_word, X_tr_char], format="csr")
    X_va_wc = sparse.hstack([X_va_word, X_va_char], format="csr")
    m2 = LinearSVC(C=1.0, class_weight="balanced", random_state=random_state).fit(X_tr_wc, train_y)
    rows.append({"split": "val", "model": "word_char_linear", "f1": float(f1_score(val_y, m2.predict(X_va_wc)))})

    dense_builder = HybridFeatureBuilder(word_max_features=1, char_max_features=1)
    X_tr_dense = dense_builder.fit_transform(pd.Series(train_texts))
    X_va_dense = dense_builder.transform(pd.Series(val_texts))
    m3 = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state).fit(X_tr_dense, train_y)
    rows.append({"split": "val", "model": "hybrid_dense_proxy", "f1": float(f1_score(val_y, m3.predict(X_va_dense)))})

    save_ablation_table(rows, out_dir / "ablation.csv", out_dir / "ablation.md")


def evaluate_models(y, prob_linear, prob_bert, prob_ensemble, threshold: float) -> dict:
    return {
        "linear_svc": classification_metrics(y, prob_linear, threshold),
        "distilbert": classification_metrics(y, prob_bert, threshold),
        "ensemble": classification_metrics(y, prob_ensemble, threshold),
    }


def run_cross_domain_validation(
    df: pd.DataFrame,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict:
    bundle = source_holdout_split(df, source_column="subject", holdout_ratio=args.cross_domain_holdout_ratio)

    feature_builder = HybridFeatureBuilder(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
    )
    X_train = feature_builder.fit_transform(bundle.train["combined_text"])
    X_val = feature_builder.transform(bundle.val["combined_text"])
    X_test = feature_builder.transform(bundle.test["combined_text"])

    y_train = bundle.train["label"].astype(int).values
    y_val = bundle.val["label"].astype(int).values
    y_test = bundle.test["label"].astype(int).values
    LOGGER.info(
        "Class counts | train(real=%d,fake=%d) val(real=%d,fake=%d) test(real=%d,fake=%d)",
        int((y_train == 0).sum()),
        int((y_train == 1).sum()),
        int((y_val == 0).sum()),
        int((y_val == 1).sum()),
        int((y_test == 0).sum()),
        int((y_test == 1).sum()),
    )
    if len(np.unique(y_val)) < 2:
        LOGGER.warning(
            "Validation split has only one class with current time split. "
            "AUC will be set to 0.5 for BERT eval steps; consider --split_method source for more stable validation."
        )

    linear_model = train_linear_svc(X_train, y_train, args.random_state)
    val_prob_linear = linear_model.predict_proba(X_val)[:, 1]
    test_prob_linear = linear_model.predict_proba(X_test)[:, 1]

    bert_model_dir = output_dir / "distilbert_cross_domain"
    if args.run_bert:
        try:
            trainer = train_distilbert(
                train_texts=bundle.train["combined_text"].values,
                train_labels=y_train,
                val_texts=bundle.val["combined_text"].values,
                val_labels=y_val,
                model_dir=bert_model_dir,
                random_state=args.random_state,
                epochs=args.bert_epochs,
                batch_size=args.bert_batch_size,
                learning_rate=args.bert_learning_rate,
                max_length=args.bert_max_length,
                use_gpu=args.bert_use_gpu,
            )
            val_prob_bert = distilbert_predict_proba(
                trainer, bundle.val["combined_text"].values, max_length=args.bert_max_length
            )
            test_prob_bert = distilbert_predict_proba(
                trainer, bundle.test["combined_text"].values, max_length=args.bert_max_length
            )
        except ModuleNotFoundError as err:
            LOGGER.error(
                "BERT dependencies are missing in cross-domain run (%s). Install with: "
                "pip install datasets transformers torch accelerate",
                err,
            )
            LOGGER.warning("Falling back to neutral BERT probabilities (0.5) for cross-domain run.")
            val_prob_bert = np.full_like(val_prob_linear, 0.5)
            test_prob_bert = np.full_like(test_prob_linear, 0.5)
    else:
        val_prob_bert = np.full_like(val_prob_linear, 0.5)
        test_prob_bert = np.full_like(test_prob_linear, 0.5)

    stacker = StackingEnsemble(random_state=args.random_state)
    stacker.fit(np.column_stack([val_prob_linear, val_prob_bert]), y_val)
    test_prob_ensemble = stacker.predict_proba(np.column_stack([test_prob_linear, test_prob_bert]))

    metrics = evaluate_models(y_test, test_prob_linear, test_prob_bert, test_prob_ensemble, args.threshold)
    comparison = build_model_comparison_table(metrics)
    comparison.to_csv(output_dir / "cross_domain_model_comparison.csv", index=False)

    return {
        "split": {
            "train": len(bundle.train),
            "val": len(bundle.val),
            "test": len(bundle.test),
            "holdout_ratio": args.cross_domain_holdout_ratio,
        },
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust fake-news training pipeline with stacking and stress tests")
    parser.add_argument("--true_csv", default="True.csv")
    parser.add_argument("--fake_csv", default="Fake.csv")
    parser.add_argument("--output_dir", default="models/research")
    parser.add_argument("--split_method", choices=["time", "source"], default="time")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--val_size", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--sample_size", type=int, default=0, help="If >0, sample this many rows before split.")
    parser.add_argument("--sample_frac", type=float, default=1.0, help="If <1.0, sample this fraction before split.")
    parser.add_argument("--word_max_features", type=int, default=150000)
    parser.add_argument("--char_max_features", type=int, default=120000)

    parser.add_argument("--run_bert", action="store_true", help="Enable DistilBERT fine-tuning")
    parser.add_argument("--bert_epochs", type=int, default=1)
    parser.add_argument("--bert_batch_size", type=int, default=8)
    parser.add_argument("--bert_learning_rate", type=float, default=2e-5)
    parser.add_argument("--bert_max_length", type=int, default=192)
    parser.add_argument("--bert_use_gpu", action="store_true", help="Prefer GPU for DistilBERT if CUDA is available.")

    parser.add_argument("--run_ablation", action="store_true")
    parser.add_argument("--run_cross_domain", action="store_true")
    parser.add_argument("--cross_domain_holdout_ratio", type=float, default=0.3)
    parser.add_argument("--robustness_sample_size", type=int, default=1200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("INFO")
    set_seed(args.random_state)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_true_fake_dataset(Path(args.true_csv), Path(args.fake_csv))
    if args.sample_size > 0:
        sample_n = min(args.sample_size, len(df))
        df = df.sample(n=sample_n, random_state=args.random_state).reset_index(drop=True)
        LOGGER.info("Using sampled dataset size: %d", len(df))
    elif args.sample_frac < 1.0:
        if not (0.0 < args.sample_frac <= 1.0):
            raise ValueError("--sample_frac must be in (0, 1].")
        df = df.sample(frac=args.sample_frac, random_state=args.random_state).reset_index(drop=True)
        LOGGER.info("Using sampled fraction %.3f -> rows=%d", args.sample_frac, len(df))

    bundle = split_dataset(
        df,
        SplitConfig(
            method=args.split_method,
            test_size=args.test_size,
            val_size=args.val_size,
            random_state=args.random_state,
            source_column="subject",
        ),
    )

    train_text = bundle.train["combined_text"]
    val_text = bundle.val["combined_text"]
    test_text = bundle.test["combined_text"]

    y_train = bundle.train["label"].astype(int).values
    y_val = bundle.val["label"].astype(int).values
    y_test = bundle.test["label"].astype(int).values

    feature_builder = HybridFeatureBuilder(
        word_max_features=args.word_max_features,
        char_max_features=args.char_max_features,
    )
    X_train = feature_builder.fit_transform(train_text)
    X_val = feature_builder.transform(val_text)
    X_test = feature_builder.transform(test_text)

    LOGGER.info("Training LinearSVC")
    linear_model = train_linear_svc(X_train, y_train, args.random_state)
    val_prob_linear = linear_model.predict_proba(X_val)[:, 1]
    test_prob_linear = linear_model.predict_proba(X_test)[:, 1]

    bert_model_dir = output_dir / "distilbert"
    if args.run_bert:
        LOGGER.info("Training DistilBERT")
        try:
            trainer = train_distilbert(
                train_texts=train_text.values,
                train_labels=y_train,
                val_texts=val_text.values,
                val_labels=y_val,
                model_dir=bert_model_dir,
                random_state=args.random_state,
                epochs=args.bert_epochs,
                batch_size=args.bert_batch_size,
                learning_rate=args.bert_learning_rate,
                max_length=args.bert_max_length,
                use_gpu=args.bert_use_gpu,
            )
            val_prob_bert = distilbert_predict_proba(trainer, val_text.values, max_length=args.bert_max_length)
            test_prob_bert = distilbert_predict_proba(trainer, test_text.values, max_length=args.bert_max_length)
        except ModuleNotFoundError as err:
            LOGGER.error(
                "BERT dependencies are missing (%s). Install with: "
                "pip install datasets transformers torch accelerate",
                err,
            )
            LOGGER.warning("Falling back to neutral BERT probabilities (0.5).")
            trainer = None
            val_prob_bert = np.full_like(val_prob_linear, 0.5)
            test_prob_bert = np.full_like(test_prob_linear, 0.5)
    else:
        LOGGER.warning("DistilBERT disabled. Use --run_bert for semantic model training.")
        trainer = None
        val_prob_bert = np.full_like(val_prob_linear, 0.5)
        test_prob_bert = np.full_like(test_prob_linear, 0.5)

    stacker = StackingEnsemble(random_state=args.random_state)
    stacker.fit(np.column_stack([val_prob_linear, val_prob_bert]), y_val)
    val_prob_ensemble = stacker.predict_proba(np.column_stack([val_prob_linear, val_prob_bert]))
    test_prob_ensemble = stacker.predict_proba(np.column_stack([test_prob_linear, test_prob_bert]))

    val_metrics = evaluate_models(y_val, val_prob_linear, val_prob_bert, val_prob_ensemble, args.threshold)
    test_metrics = evaluate_models(y_test, test_prob_linear, test_prob_bert, test_prob_ensemble, args.threshold)

    comparison = build_model_comparison_table(test_metrics)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    plot_calibration_curves(
        y_true=y_test,
        probs={
            "LinearSVC": test_prob_linear,
            "DistilBERT": test_prob_bert,
            "Ensemble": test_prob_ensemble,
        },
        out_path=output_dir / "calibration_curve.png",
    )
    plot_reliability_diagram(
        y_true=y_test,
        probs={
            "LinearSVC": test_prob_linear,
            "DistilBERT": test_prob_bert,
            "Ensemble": test_prob_ensemble,
        },
        out_path=output_dir / "reliability_diagram.png",
    )

    def linear_predict_fn(texts: list[str]) -> np.ndarray:
        X = feature_builder.transform(pd.Series(texts))
        return linear_model.predict_proba(X)[:, 1]

    def bert_predict_fn(texts: list[str]) -> np.ndarray:
        if args.run_bert and trainer is not None:
            return distilbert_predict_proba(trainer, texts, max_length=args.bert_max_length)
        return np.full(len(texts), 0.5, dtype=np.float64)

    def ensemble_predict_fn(texts: list[str]) -> np.ndarray:
        p1 = linear_predict_fn(texts)
        p2 = bert_predict_fn(texts)
        return stacker.predict_proba(np.column_stack([p1, p2]))

    robustness = {
        "linear_svc": run_robustness_suite(
            texts=test_text.tolist(),
            y_true=y_test,
            predict_proba_fn=linear_predict_fn,
            threshold=args.threshold,
            sample_size=args.robustness_sample_size,
            random_state=args.random_state,
        ),
        "distilbert": run_robustness_suite(
            texts=test_text.tolist(),
            y_true=y_test,
            predict_proba_fn=bert_predict_fn,
            threshold=args.threshold,
            sample_size=args.robustness_sample_size,
            random_state=args.random_state,
        ),
        "ensemble": run_robustness_suite(
            texts=test_text.tolist(),
            y_true=y_test,
            predict_proba_fn=ensemble_predict_fn,
            threshold=args.threshold,
            sample_size=args.robustness_sample_size,
            random_state=args.random_state,
        ),
    }
    save_robustness_report(robustness, output_dir / "robustness_report.json")

    cross_domain = None
    if args.run_cross_domain:
        LOGGER.info("Running cross-domain validation on unseen sources")
        cross_domain = run_cross_domain_validation(df, args, output_dir)
        save_json(cross_domain, output_dir / "cross_domain_report.json")

    if args.run_ablation:
        run_ablation(train_text.values, y_train, val_text.values, y_val, output_dir, args.random_state)

    best_model = max(val_metrics.keys(), key=lambda k: val_metrics[k]["f1"])
    artifact = {
        "version": "3.0-robust",
        "random_state": args.random_state,
        "threshold": args.threshold,
        "split_method": args.split_method,
        "label_map": {0: "REAL", 1: "FAKE"},
        "feature_artifacts": feature_builder.export(),
        "linear_svc": linear_model,
        "bert_model_dir": str(bert_model_dir),
        "stacking_meta_model": stacker.meta_model,
        "stacking_constant_prob": stacker.constant_prob,
        "stacking_use_average": stacker.use_average,
        "best_model": best_model,
    }
    joblib.dump(artifact, output_dir / "final_model.joblib")

    report = {
        "dataset_rows": int(len(df)),
        "split": {
            "method": args.split_method,
            "train": int(len(bundle.train)),
            "val": int(len(bundle.val)),
            "test": int(len(bundle.test)),
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "best_model": best_model,
        "cross_domain": cross_domain,
    }
    save_json(report, output_dir / "metrics_report.json")

    LOGGER.info("Best model by validation F1: %s", best_model)
    LOGGER.info("Saved artifact: %s", output_dir / "final_model.joblib")


if __name__ == "__main__":
    main()
