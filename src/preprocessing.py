import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


LOGGER = logging.getLogger(__name__)


LEAKAGE_PATTERNS = [
    re.compile(r"^\s*\(?reuters\)?\s*[-:]\s*", re.IGNORECASE),
    re.compile(r"^\s*\w+\s*\(reuters\)\s*[-:]\s*", re.IGNORECASE),
    re.compile(r"^\s*\([a-z\-\s]+\)\s*(?:-|\u2013|\u2014)\s*"),
    re.compile(r"\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b"),
    re.compile(r"^\s*(ap|associated press|afp|cnn|bbc|fox news|nytimes|the guardian)\s*[-:]\s*", re.IGNORECASE),
    re.compile(r"\b(file photo|exclusive|breaking|updated)\b\s*[:\-]?\s*", re.IGNORECASE),
]


@dataclass
class SplitConfig:
    method: Literal["time", "source"] = "time"
    test_size: float = 0.2
    val_size: float = 0.1
    time_column: str = "date"
    source_column: str = "subject"
    random_state: int = 42


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def set_seed(seed: int) -> None:
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def load_csv_any_encoding(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]
    last_error = None
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError as err:
            last_error = err
    raise ValueError(f"Could not decode {path}. Last error: {last_error}")


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_source_leakage(text: str) -> str:
    clean = _normalize_text(text)
    for pattern in LEAKAGE_PATTERNS:
        clean = pattern.sub("", clean)

    # Remove repetitive agency/location lead-ins and newsroom formatting.
    clean = re.sub(r"^\s*[A-Z][A-Z\s\.,]{2,40}\s*(?:-|\u2013|\u2014)\s*", "", clean)
    clean = re.sub(r"^\s*[A-Za-z\.\s]{2,30}\s*\(\w+\)\s*(?:-|\u2013|\u2014)\s*", "", clean)
    clean = re.sub(r"\s*\|\s*([A-Za-z ]{2,25})\s*$", "", clean)
    return clean.strip()


def build_combined_text(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["title", "text", "subject", "date"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    df["title_clean"] = df["title"].map(strip_source_leakage)
    df["text_clean"] = df["text"].map(strip_source_leakage)
    df["subject_clean"] = df["subject"].map(_normalize_text)
    df["date_clean"] = df["date"].map(_normalize_text)

    df["combined_text"] = (
        "title: "
        + df["title_clean"]
        + " text: "
        + df["text_clean"]
        + " subject: "
        + df["subject_clean"]
        + " date: "
        + df["date_clean"]
    ).str.lower()

    return df


def parse_dates_with_fallback(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().all():
        LOGGER.warning("All dates failed to parse; defaulting to pseudo-ordering.")
        parsed = pd.to_datetime(pd.RangeIndex(start=0, stop=len(series), step=1), unit="D")
    return parsed


def load_true_fake_dataset(true_csv: Path, fake_csv: Path) -> pd.DataFrame:
    true_df = load_csv_any_encoding(true_csv)
    fake_df = load_csv_any_encoding(fake_csv)

    true_df["label"] = 0
    fake_df["label"] = 1

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = build_combined_text(df)
    df = df.dropna(subset=["combined_text", "label"])
    df = df.drop_duplicates(subset=["combined_text", "label"]).reset_index(drop=True)

    if "date" not in df.columns:
        df["date"] = ""
    df["date_parsed"] = parse_dates_with_fallback(df["date"])

    if "subject" not in df.columns:
        df["subject"] = "unknown"

    return df


def _time_split(df: pd.DataFrame, config: SplitConfig) -> DatasetBundle:
    ordered = df.sort_values("date_parsed", ascending=True).reset_index(drop=True)
    n = len(ordered)

    test_n = int(round(n * config.test_size))
    val_n = int(round(n * config.val_size))
    train_n = n - test_n - val_n

    if train_n <= 0 or val_n <= 0 or test_n <= 0:
        raise ValueError("Invalid split sizes. Adjust val_size/test_size.")

    train = ordered.iloc[:train_n].reset_index(drop=True)
    val = ordered.iloc[train_n : train_n + val_n].reset_index(drop=True)
    test = ordered.iloc[train_n + val_n :].reset_index(drop=True)
    return DatasetBundle(train=train, val=val, test=test)


def _source_split(df: pd.DataFrame, config: SplitConfig) -> DatasetBundle:
    groups = df[config.source_column].fillna("unknown").astype(str).values
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_state)
    train_val_idx, test_idx = next(splitter.split(df, groups=groups))

    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    groups_tv = train_val[config.source_column].fillna("unknown").astype(str).values
    val_rel = config.val_size / max(1e-9, (1.0 - config.test_size))
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_rel, random_state=config.random_state)
    train_idx, val_idx = next(splitter_val.split(train_val, groups=groups_tv))

    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)
    return DatasetBundle(train=train, val=val, test=test)


def split_dataset(df: pd.DataFrame, config: SplitConfig) -> DatasetBundle:
    if config.method == "time":
        bundle = _time_split(df, config)
    elif config.method == "source":
        bundle = _source_split(df, config)
    else:
        raise ValueError(f"Unsupported split method: {config.method}")

    LOGGER.info(
        "Split=%s | train=%d val=%d test=%d",
        config.method,
        len(bundle.train),
        len(bundle.val),
        len(bundle.test),
    )
    return bundle


def source_holdout_split(
    df: pd.DataFrame,
    source_column: str = "subject",
    holdout_ratio: float = 0.3,
    min_sources: int = 2,
) -> DatasetBundle:
    """Hold out complete source domains for cross-domain validation."""
    sources = df[source_column].fillna("unknown").astype(str)
    source_counts = sources.value_counts()
    unique_sources = list(source_counts.index)

    if len(unique_sources) < min_sources:
        raise ValueError(f"Need at least {min_sources} unique sources for source holdout split.")

    holdout_n = max(1, int(round(len(unique_sources) * holdout_ratio)))
    holdout_sources = set(unique_sources[-holdout_n:])

    is_holdout = sources.isin(holdout_sources)
    test = df[is_holdout].reset_index(drop=True)
    train_val = df[~is_holdout].reset_index(drop=True)

    # Validation uses an internal source split over remaining domains.
    internal_cfg = SplitConfig(method="source", test_size=0.2, val_size=0.2, source_column=source_column)
    internal_bundle = _source_split(train_val, internal_cfg)
    train = internal_bundle.train
    val = internal_bundle.val

    LOGGER.info(
        "Cross-domain holdout | holdout_sources=%s | train=%d val=%d test=%d",
        sorted(holdout_sources),
        len(train),
        len(val),
        len(test),
    )
    return DatasetBundle(train=train, val=val, test=test)
