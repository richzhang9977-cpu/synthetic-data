from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_utils import PAPER_BEST_FEATURES as FEATURES
from temporal_utils import TARGET, load_sessions


RAW_STEP = 10
SEQUENCE_LENGTH = 30
SEQUENCE_STRIDE = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="F:/synthetic/dataset/indoor")
    parser.add_argument("--output-dir", default="F:/synthetic/dataset/processed_every10")
    return parser.parse_args()


def sampled_rows(sessions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for session_id, frame in sessions.items():
        sampled = frame.iloc[::RAW_STEP].copy().reset_index(drop=True)
        sampled.insert(0, "session_id", str(session_id))
        sampled.insert(1, "step", np.arange(len(sampled), dtype=int))
        parts.append(sampled[["session_id", "step", "frame", "Timestamp", *FEATURES, TARGET]])
    return pd.concat(parts, ignore_index=True)


def fixed_label_sequences(rows: pd.DataFrame, split: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for session_id, session in rows.groupby("session_id", sort=False):
        session = session.sort_values("step", kind="stable").reset_index(drop=True)
        for start in range(0, len(session) - SEQUENCE_LENGTH + 1, SEQUENCE_STRIDE):
            window = session.iloc[start : start + SEQUENCE_LENGTH]
            if window[TARGET].nunique() != 1:
                continue
            sequence = window.copy()
            sequence.insert(0, "SequenceID", f"{split}:{session_id}:{start}")
            sequence["Step"] = np.arange(SEQUENCE_LENGTH, dtype=int)
            parts.append(
                sequence[
                    [
                        "SequenceID",
                        "Step",
                        "session_id",
                        "step",
                        "frame",
                        "Timestamp",
                        *FEATURES,
                        TARGET,
                    ]
                ]
            )
    if not parts:
        raise ValueError(f"No eligible {split} sequences were constructed")
    return pd.concat(parts, ignore_index=True)


def span_summary(sequences: pd.DataFrame) -> dict[str, float]:
    timestamps = sequences.assign(Timestamp=pd.to_datetime(sequences["Timestamp"], errors="raise"))
    spans = timestamps.groupby("SequenceID")["Timestamp"].agg(
        lambda values: (values.iloc[-1] - values.iloc[0]).total_seconds()
    )
    return {
        "median_seconds": float(spans.median()),
        "mean_seconds": float(spans.mean()),
        "p05_seconds": float(spans.quantile(0.05)),
        "p95_seconds": float(spans.quantile(0.95)),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {
        "raw_step": RAW_STEP,
        "sequence_length": SEQUENCE_LENGTH,
        "sequence_stride": SEQUENCE_STRIDE,
        "fixed_label_only": True,
        "numeric_frame_sort": True,
        "cross_session_windows": False,
        "features": FEATURES,
        "splits": {},
    }
    for split in ("train", "test"):
        sessions = load_sessions(args.dataset_root, split)
        rows = sampled_rows(sessions)
        sequences = fixed_label_sequences(rows, split)
        rows.to_csv(output_dir / f"{split}.csv", index=False)
        sequences.to_csv(output_dir / f"{split}_sequences.csv", index=False)
        sequence_labels = sequences.groupby("SequenceID")[TARGET].first().astype(int)
        metadata["splits"][split] = {
            "sessions": int(rows["session_id"].nunique()),
            "sampled_rows": int(len(rows)),
            "sequences": int(sequences["SequenceID"].nunique()),
            "sequence_label_counts": {
                str(label): int(count)
                for label, count in sequence_labels.value_counts().sort_index().items()
            },
            "physical_span": span_summary(sequences),
        }
        print(
            f"{split}: sessions={rows['session_id'].nunique()} "
            f"sampled_rows={len(rows):,} sequences={sequences['SequenceID'].nunique():,}",
            flush=True,
        )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
