"""Min-max normalization for audio features across the entire catalog.

Audio features span different native ranges (tempo ~40-220 vs. valence 0-1).
Without normalization, larger-range features would dominate the audio
sub-vector. We compute min/max once per feature over the whole catalog and
persist it, so query-time normalization uses the same bounds."""

import json
from pathlib import Path

import numpy as np

from engine import config


def compute_stats(rows: list[dict]) -> dict:
    """Compute min/max for each AUDIO_FEATURE over the given rows.

    Returns: {feature: {"min": float, "max": float}, ...}"""
    stats = {feat: {"min": float("inf"), "max": float("-inf")} for feat in config.AUDIO_FEATURES}

    for row in rows:
        for feat in config.AUDIO_FEATURES:
            value = float(row[feat])
            stats[feat]["min"] = min(stats[feat]["min"], value)
            stats[feat]["max"] = max(stats[feat]["max"], value)

    return stats


def apply(row: dict, stats: dict) -> np.ndarray:
    """Normalize row's AUDIO_FEATURES into [0,1] using stats.

    Returns: ndarray of shape (9,) in AUDIO_FEATURES order.
    For zero-variance features (min == max), returns 0.5 instead of NaN."""
    normalized = []
    for feat in config.AUDIO_FEATURES:
        value = float(row[feat])
        feat_min = stats[feat]["min"]
        feat_max = stats[feat]["max"]

        if feat_max == feat_min:
            normalized.append(0.5)
        else:
            normalized.append((value - feat_min) / (feat_max - feat_min))

    return np.array(normalized, dtype=np.float32)


def save_stats(stats: dict, path: Path) -> None:
    """Persist normalization stats to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(stats, f, indent=2)


def load_stats(path: Path) -> dict:
    """Load normalization stats from JSON."""
    with path.open() as f:
        return json.load(f)
