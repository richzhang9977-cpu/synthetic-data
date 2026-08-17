from __future__ import annotations

import json
from pathlib import Path

from audit_direct_prompt_ablations import audit, sequence_signatures


ROOT = Path("F:/synthetic")
INPUTS = {
    "RealTrain": ROOT / "dataset/processed_every10/train_sequences.csv",
    "DenseNaive": ROOT / "llm_synthetic/every10/direct_naive.csv",
    "DenseControlled": ROOT
    / "llm_synthetic/every10/direct_controlled_nocopy.csv",
    "SparseNaive": ROOT / "llm_synthetic/every10/sparse_naive_events_b10.csv",
    "SparseControlledV5": ROOT
    / "llm_synthetic/every10/approach_a_v5_events_b10.csv",
}
OUTPUT = ROOT / "results/every10/sparse_prompt_2x2_audit.json"


def main() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing 2x2 ablation inputs: {missing}")

    real_signatures = set(sequence_signatures(INPUTS["RealTrain"]))
    reports = {}
    for name, path in INPUTS.items():
        report = audit(path)
        if name != "RealTrain":
            signatures = sequence_signatures(path)
            matches = sum(signature in real_signatures for signature in signatures)
            report["sequences_matching_real_train"] = int(matches)
            report["sequence_match_real_train_rate"] = float(
                matches / len(signatures)
            )
        reports[name] = report

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"datasets": reports}, indent=2), encoding="utf-8")
    print(json.dumps({"datasets": reports}, indent=2))
    print(f"Saved audit to {OUTPUT}")


if __name__ == "__main__":
    main()
