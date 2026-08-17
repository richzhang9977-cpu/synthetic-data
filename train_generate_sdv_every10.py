from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from rdt.transformers.numerical import GaussianNormalizer

from temporal_utils import PAPER_BEST_FEATURES as FEATURES
from temporal_utils import TARGET


SEQUENCE_LENGTH = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-train",
        default="F:/synthetic/dataset/processed_every10/train_sequences.csv",
    )
    parser.add_argument(
        "--output-dir", default="F:/synthetic/llm_synthetic/every10/sdv"
    )
    parser.add_argument("--methods", nargs="+", choices=("TVAE", "CTGAN"), default=("TVAE", "CTGAN"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--sequences", type=int, default=1_680)
    return parser.parse_args()


def validate_sequences(frame: pd.DataFrame) -> None:
    required = {"SequenceID", "Step", *FEATURES, TARGET}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Shared sequence input is missing columns: {sorted(missing)}")
    lengths = frame.groupby("SequenceID").size()
    mixed = frame.groupby("SequenceID")[TARGET].nunique()
    steps_ok = frame.groupby("SequenceID")["Step"].apply(
        lambda values: np.array_equal(np.sort(values.to_numpy(dtype=int)), np.arange(SEQUENCE_LENGTH))
    )
    if not (lengths == SEQUENCE_LENGTH).all() or not (mixed == 1).all() or not steps_ok.all():
        raise ValueError("SDV requires fixed-label, complete 30-step sequences")


def to_wide(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for _, sequence in frame.groupby("SequenceID", sort=False):
        sequence = sequence.sort_values("Step", kind="stable")
        row: dict[str, float | int] = {TARGET: int(sequence[TARGET].iloc[0])}
        for step, (_, values) in enumerate(sequence.iterrows()):
            for feature in FEATURES:
                row[f"{feature}__t{step:02d}"] = float(values[feature])
        rows.append(row)
    return pd.DataFrame(rows)


def to_long(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[list[float | int]] = []
    for sequence_id, (_, wide_row) in enumerate(frame.iterrows()):
        label = int(wide_row[TARGET])
        for step in range(SEQUENCE_LENGTH):
            rows.append(
                [sequence_id, step, *[float(wide_row[f"{feature}__t{step:02d}"]) for feature in FEATURES], label]
            )
    return pd.DataFrame(rows, columns=["SequenceID", "Step", *FEATURES, TARGET])


def build_synthesizer(method: str, metadata: SingleTableMetadata, epochs: int):
    common = {
        "metadata": metadata,
        "epochs": epochs,
        "enforce_min_max_values": True,
        "enforce_rounding": False,
        "cuda": True,
        "verbose": True,
    }
    if method == "TVAE":
        return TVAESynthesizer(**common)
    return CTGANSynthesizer(**common)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences = pd.read_csv(args.real_train)
    validate_sequences(sequences)
    wide = to_wide(sequences)
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(wide)
    metadata.update_column(TARGET, sdtype="categorical")

    for method in args.methods:
        print(f"Training {method} on {len(wide):,} real sequences...", flush=True)
        started = time.time()
        synthesizer = build_synthesizer(method, metadata, args.epochs)
        synthesizer.auto_assign_transformers(wide)
        synthesizer.update_transformers(
            {
                column: GaussianNormalizer(
                    distribution="norm", enforce_min_max_values=True
                )
                for column in wide.columns
                if column != TARGET
            }
        )
        synthesizer.fit(wide)
        generated_wide = synthesizer.sample(num_rows=args.sequences)
        generated_wide[TARGET] = pd.to_numeric(
            generated_wide[TARGET], errors="raise"
        ).astype(int)
        counts = generated_wide[TARGET].value_counts().sort_index()
        generated = to_long(generated_wide)
        method_dir = output_dir / method.lower()
        method_dir.mkdir(parents=True, exist_ok=True)
        generated.to_csv(method_dir / "synthetic.csv", index=False)
        synthesizer.save(filepath=str(method_dir / "model.pkl"))
        audit = {
            "method": method,
            "features": FEATURES,
            "training_sequences": int(len(wide)),
            "generated_sequences": int(len(generated_wide)),
            "generated_rows": int(len(generated)),
            "generated_label_counts": {
                str(int(label)): int(count) for label, count in counts.items()
            },
            "sequence_length": SEQUENCE_LENGTH,
            "epochs": args.epochs,
            "elapsed_seconds": time.time() - started,
            "shared_preprocessing": "four paper-best features; every 10 raw observations; fixed-label 30-step session windows",
        }
        (method_dir / "metadata.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
