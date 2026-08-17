from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_utils import PAPER_BEST_FEATURES as FEATURES
from temporal_utils import TARGET


ROOT = Path("F:/synthetic")
INPUTS = {
    "RealTrain": ROOT / "dataset/processed_every10/train_sequences.csv",
    "DirectNaive": ROOT / "llm_synthetic/every10/direct_naive.csv",
    "DirectControlledCopyBug": ROOT / "llm_synthetic/every10/direct_controlled.csv",
    "DirectControlled": ROOT
    / "llm_synthetic/every10/direct_controlled_nocopy.csv",
    "DeepSeekV5B10": ROOT
    / "llm_synthetic/every10/approach_a_v5_events_b10.csv",
}
METRICS = {
    "DirectNaive": ROOT / "results/every10/direct_naive_generation_metrics.json",
    "DirectControlledCopyBug": ROOT
    / "results/every10/direct_controlled_generation_metrics.json",
    "DirectControlled": ROOT
    / "results/every10/direct_controlled_nocopy_generation_metrics.json",
}
OUTPUT = ROOT / "results/every10/direct_prompt_ablation_audit.json"


def temporal_stats(data: pd.DataFrame) -> dict:
    deltas = []
    for _, sequence in data.groupby("SequenceID", sort=False):
        sequence = sequence.sort_values("Step")
        deltas.append(np.diff(sequence[FEATURES].to_numpy(dtype=float), axis=0))
    delta = np.concatenate(deltas, axis=0)
    result = {}
    for index, feature in enumerate(FEATURES):
        absolute = np.abs(delta[:, index])
        nonzero = absolute[absolute > 1e-12]
        result[feature] = {
            "change_probability_per_step": float((absolute > 1e-12).mean()),
            "median_nonzero_abs_delta": float(np.median(nonzero)) if len(nonzero) else 0.0,
            "p95_nonzero_abs_delta": (
                float(np.percentile(nonzero, 95)) if len(nonzero) else 0.0
            ),
        }
    return result


def audit(path: Path) -> dict:
    data = pd.read_csv(path)
    grouped = data.groupby("SequenceID", sort=False)
    signatures = grouped[FEATURES].apply(
        lambda frame: tuple(map(tuple, frame.to_numpy(dtype=float)))
    )
    lengths = grouped.size()
    labels = grouped[TARGET].first().astype(int)
    return {
        "path": str(path),
        "rows": int(len(data)),
        "sequences": int(grouped.ngroups),
        "sequence_lengths": sorted(map(int, lengths.unique())),
        "mixed_label_sequences": int((grouped[TARGET].nunique() != 1).sum()),
        "missing_values": int(data[FEATURES].isna().sum().sum()),
        "non_finite_values": int((~np.isfinite(data[FEATURES].to_numpy(float))).sum()),
        "label_counts": {
            str(int(label)): int(count)
            for label, count in labels.value_counts().sort_index().items()
        },
        "exact_duplicate_sequences": int(signatures.duplicated().sum()),
        "exact_duplicate_sequence_rate": float(signatures.duplicated().mean()),
        "temporal": temporal_stats(data),
    }


def generation_summary(path: Path) -> dict:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    totals = metrics["totals"]
    accepted = int(totals["accepted_sequences"])
    returned = int(totals["returned_objects"])
    return {
        "attempts": int(totals["attempts"]),
        "returned_objects": returned,
        "accepted_sequences": accepted,
        "object_acceptance_rate": float(accepted / returned) if returned else 0.0,
        "prompt_tokens": int(totals["prompt_tokens"]),
        "completion_tokens": int(totals["completion_tokens"]),
        "total_tokens": int(totals["total_tokens"]),
        "tokens_per_accepted_sequence": float(totals["total_tokens"] / accepted),
        "elapsed_seconds": float(totals["elapsed_seconds"]),
        "rejections": totals["rejections"],
    }


def sequence_signatures(path: Path) -> pd.Series:
    data = pd.read_csv(path)
    return data.groupby("SequenceID", sort=False)[FEATURES].apply(
        lambda frame: tuple(map(tuple, frame.to_numpy(dtype=float)))
    )


def main() -> None:
    real_signatures = set(sequence_signatures(INPUTS["RealTrain"]))
    dataset_reports = {name: audit(path) for name, path in INPUTS.items()}
    for name, path in INPUTS.items():
        if name == "RealTrain":
            continue
        signatures = sequence_signatures(path)
        dataset_reports[name]["sequences_matching_real_train"] = int(
            sum(signature in real_signatures for signature in signatures)
        )
        dataset_reports[name]["unique_sequences_matching_real_train"] = int(
            sum(signature in real_signatures for signature in set(signatures))
        )
    report = {
        "datasets": dataset_reports,
        "generation": {
            name: generation_summary(path) for name, path in METRICS.items()
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved audit to {OUTPUT}")


if __name__ == "__main__":
    main()
