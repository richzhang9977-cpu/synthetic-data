from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_utils import PAPER_BEST_FEATURES as FEATURES
from temporal_utils import TARGET


ROOT = Path("F:/synthetic")
EXPLICIT = {
    "DeepSeekV5B10": ROOT
    / "llm_synthetic/every10/approach_a_v5_events_b10.csv",
    "SparseNaiveB10": ROOT
    / "llm_synthetic/every10/sparse_naive_events_b10.csv",
    "TVAE": ROOT / "llm_synthetic/every10/sdv/tvae/synthetic.csv",
    "CTGAN": ROOT / "llm_synthetic/every10/sdv/ctgan/synthetic.csv",
}
REAL = ROOT / "dataset/processed_every10/train_sequences.csv"
OUTPUT = ROOT / "results/every10/generation_audit.json"


def feature_ranges(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        feature: {
            "min": float(frame[feature].min()),
            "max": float(frame[feature].max()),
        }
        for feature in FEATURES
    }


def label_counts(values: pd.Series) -> dict[str, int]:
    return {
        str(int(label)): int(count)
        for label, count in values.value_counts().sort_index().items()
    }


def explicit_audit(path: Path, real_min: pd.Series, real_max: pd.Series) -> dict:
    frame = pd.read_csv(path)
    required = {"SequenceID", "Step", *FEATURES, TARGET}
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"{path} missing {missing_columns}")
    groups = frame.groupby("SequenceID", sort=False)
    lengths = groups.size()
    sequence_labels = groups[TARGET].first().astype(int)
    signatures: list[tuple[int, bytes]] = []
    for _, sequence in groups:
        sequence = sequence.sort_values("Step", kind="stable")
        signatures.append(
            (
                int(sequence[TARGET].iloc[0]),
                np.round(sequence[FEATURES].to_numpy(dtype=float), 7).tobytes(),
            )
        )
    unique_signatures = len(set(signatures))
    below = (frame[FEATURES] < real_min).sum()
    above = (frame[FEATURES] > real_max).sum()
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "sequences": int(len(lengths)),
        "sequence_lengths": sorted(int(value) for value in lengths.unique()),
        "invalid_step_sequences": int(
            groups["Step"].apply(
                lambda values: not np.array_equal(
                    np.sort(values.to_numpy(dtype=int)), np.arange(30)
                )
            ).sum()
        ),
        "mixed_label_sequences": int(groups[TARGET].nunique().gt(1).sum()),
        "missing_values": int(frame[[*FEATURES, TARGET]].isna().sum().sum()),
        "non_finite_values": int(
            (~np.isfinite(frame[FEATURES].to_numpy(dtype=float))).sum()
        ),
        "label_sequence_counts": label_counts(sequence_labels),
        "exact_duplicate_sequences": int(len(signatures) - unique_signatures),
        "exact_duplicate_sequence_rate": float(
            (len(signatures) - unique_signatures) / len(signatures)
        ),
        "values_below_real_train_min": {
            feature: int(below[feature]) for feature in FEATURES
        },
        "values_above_real_train_max": {
            feature: int(above[feature]) for feature in FEATURES
        },
        "feature_ranges": feature_ranges(frame),
    }


def main() -> None:
    real = pd.read_csv(REAL)
    real_min = real[FEATURES].min()
    real_max = real[FEATURES].max()
    audit = {
        "protocol": {
            "features": FEATURES,
            "raw_step": 10,
            "sequence_length": 30,
            "real_training_sequences": int(real["SequenceID"].nunique()),
            "real_training_rows": int(len(real)),
        },
        "methods": {
            name: explicit_audit(path, real_min, real_max)
            for name, path in EXPLICIT.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
