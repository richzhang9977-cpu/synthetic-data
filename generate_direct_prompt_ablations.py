"""Direct-output DeepSeek ablations on the shared every-10 protocol.

The two modes deliberately use the same 30-step, four-feature output schema:

* direct-naive: five same-session/same-label dense examples, with no empirical
  dynamics statistics or hand-written temporal rules.
* direct-controlled: the same examples plus the session dynamics and global
  safety information used by Approach A v5.

Both modes write evaluator-ready long-form CSV files.  Network calls are made
only with --execute; otherwise prompt previews are produced.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from generate_synthetic_a_v5_events import empirical_stats, session_dynamics_stats
from temporal_utils import PAPER_BEST_FEATURES as FEATURES
from temporal_utils import TARGET, proportional_label_counts


SHORT = {
    "Wrist_Skin_Temperature": "wrist",
    "GSR": "gsr",
    "Ambient_Temperature": "ambient",
    "Ambient_Humidity": "humidity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("direct-naive", "direct-controlled"),
        required=True,
    )
    parser.add_argument(
        "--real-train",
        default="F:/synthetic/dataset/processed_every10/train_sequences.csv",
    )
    parser.add_argument("--output")
    parser.add_argument("--preview")
    parser.add_argument("--metrics")
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--total-sequences", type=int, default=1_680)
    parser.add_argument(
        "--sequences-per-label",
        type=int,
        help="Balanced small-run override; generates this many per label.",
    )
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--exemplars", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    stem = args.mode.replace("-", "_")
    if args.output is None:
        args.output = f"F:/synthetic/llm_synthetic/every10/{stem}.csv"
    if args.preview is None:
        args.preview = f"F:/synthetic/results/every10/{stem}_prompt_preview.txt"
    if args.metrics is None:
        args.metrics = f"F:/synthetic/results/every10/{stem}_generation_metrics.json"
    return args


def dense_example(window: pd.DataFrame) -> dict:
    values = window[FEATURES].to_numpy(dtype=float)
    return {
        "sequence_id": str(window["SequenceID"].iloc[0]),
        "label": int(window[TARGET].iloc[0]),
        "values": [[round(float(value), 7) for value in row] for row in values],
    }


def collect_examples(real: pd.DataFrame, length: int) -> dict[int, dict[str, list[dict]]]:
    examples = {label: {} for label in range(-3, 4)}
    for _, window in real.groupby("SequenceID", sort=False):
        window = window.sort_values("Step")
        labels = window[TARGET].to_numpy(dtype=int)
        if len(window) != length or not np.all(labels == labels[0]):
            continue
        session_id = str(window["session_id"].iloc[0])
        label = int(labels[0])
        examples[label].setdefault(session_id, []).append(dense_example(window))
    return examples


def sample_same_session_examples(
    candidates_by_session: dict[str, list[dict]],
    count: int,
    seed: int,
    label: int,
    attempt: int,
) -> tuple[str, list[dict]]:
    """Deterministically pair prompts across both direct-output modes."""
    eligible = sorted(
        (
            (session_id, session_examples)
            for session_id, session_examples in candidates_by_session.items()
            if len(session_examples) >= count
        ),
        key=lambda item: item[0],
    )
    if not eligible:
        largest = max((len(items) for items in candidates_by_session.values()), default=0)
        raise ValueError(
            f"No single session contains {count} eligible examples; largest has {largest}"
        )
    local_seed = np.random.SeedSequence([seed, label + 3, attempt])
    rng = np.random.default_rng(local_seed)
    session_id, session_examples = eligible[int(rng.integers(len(eligible)))]
    selected = rng.choice(len(session_examples), count, replace=False)
    return session_id, [session_examples[index] for index in selected]


def compact_examples(examples: list[dict]) -> list[dict]:
    return [
        {"label": example["label"], "values": example["values"]}
        for example in examples
    ]


def make_prompt(
    mode: str,
    label: int,
    count: int,
    examples: list[dict],
    session_stats: dict,
    global_stats: dict,
    lower: dict,
    upper: dict,
    length: int,
) -> str:
    bounds = {
        SHORT[feature]: [round(float(lower[feature]), 7), round(float(upper[feature]), 7)]
        for feature in FEATURES
    }
    common = f"""Generate {count} synthetic thermal-comfort sensor sequences as a JSON array.

Output requirements:
- Every sequence has exactly {length} observations and fixed label={label}.
- Each observation directly contains four numeric sensor readings.
- Feature order is exactly [wrist, gsr, ambient, humidity].
- Consecutive observations are every 10th numerically ordered raw record.
- A {length}-step sequence covers approximately 10.7 seconds.
- Do not copy any training example exactly.
- Return JSON only; no Markdown or explanation.

Schema for each array element:
{{"label": {label}, "values": [[wrist, gsr, ambient, humidity], ... exactly {length} rows]}}
"""
    if mode == "direct-naive":
        return common + f"""
Use the following same-participant-session, same-label training examples to infer
the dataset's value scale, cross-feature relationships, and temporal behavior.

Training-only examples:
{json.dumps(compact_examples(examples), ensure_ascii=False, separators=(",", ":"))}
"""

    global_max_delta = {
        SHORT[feature]: global_stats[SHORT[feature]]["max_abs_delta"]
        for feature in FEATURES
    }
    return common + f"""
Time-series rules:
- Sensor readings are piecewise constant with sparse updates, not smooth interpolation.
- Preserve realistic dependence among wrist, gsr, ambient, and humidity.
- Never exceed the supplied feature bounds.
- Never exceed the supplied maximum change between two adjacent observations.
- Ambient updates are normally 0.1 C and at most 0.3 C.
- Humidity updates are integer steps of 1, rarely 2.

Current-session dynamics:
{json.dumps(session_stats, ensure_ascii=False, indent=2)}

Global safety limits - feature bounds:
{json.dumps(bounds, ensure_ascii=False)}

Global safety limits - maximum allowed adjacent change:
{json.dumps(global_max_delta, ensure_ascii=False, indent=2)}

Training-only examples:
{json.dumps(compact_examples(examples), ensure_ascii=False, separators=(",", ":"))}
"""


def extract_json(text: str) -> list:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return value if isinstance(value, list) else []


def validate_direct(
    objects: list,
    mode: str,
    target_label: int,
    start_sequence_id: int,
    length: int,
    global_stats: dict,
    lower: dict,
    upper: dict,
) -> tuple[list[list], int, dict[str, int]]:
    rows: list[list] = []
    sequence_id = start_sequence_id
    reasons = {
        "wrong_label": 0,
        "wrong_shape": 0,
        "non_finite": 0,
        "out_of_bounds": 0,
        "excessive_delta": 0,
        "malformed": 0,
    }
    lo = np.array([lower[feature] for feature in FEATURES], dtype=float)
    hi = np.array([upper[feature] for feature in FEATURES], dtype=float)
    max_delta = np.array(
        [global_stats[SHORT[feature]]["max_abs_delta"] for feature in FEATURES],
        dtype=float,
    )
    for obj in objects:
        try:
            if int(obj["label"]) != target_label:
                reasons["wrong_label"] += 1
                continue
            values = np.asarray(obj["values"], dtype=float)
            if values.shape != (length, len(FEATURES)):
                reasons["wrong_shape"] += 1
                continue
            if not np.isfinite(values).all():
                reasons["non_finite"] += 1
                continue
            if np.any(values < lo) or np.any(values > hi):
                reasons["out_of_bounds"] += 1
                continue
            if mode == "direct-controlled":
                if np.any(np.abs(np.diff(values, axis=0)) > max_delta + 1e-9):
                    reasons["excessive_delta"] += 1
                    continue
            for step, vector in enumerate(values):
                rows.append([sequence_id, step, *vector.tolist(), target_label])
            sequence_id += 1
        except (KeyError, TypeError, ValueError, OverflowError):
            reasons["malformed"] += 1
    return rows, sequence_id, reasons


def usage_dict(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    return {
        "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
        "total_tokens": int(getattr(usage, "total_tokens", 0) or 0),
    }


def save_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    real = pd.read_csv(args.real_train)
    required = {"SequenceID", "Step", "session_id", *FEATURES, TARGET}
    missing = required - set(real.columns)
    if missing:
        raise ValueError(f"Shared sequence input is missing columns: {sorted(missing)}")
    lengths = real.groupby("SequenceID").size()
    mixed = real.groupby("SequenceID")[TARGET].nunique()
    if not (lengths == args.sequence_length).all() or not (mixed == 1).all():
        raise ValueError("Shared input must contain fixed-label 30-step sequences")

    global_stats, lower, upper = empirical_stats(real)
    session_stats = session_dynamics_stats(real, global_stats)
    examples = collect_examples(real, args.sequence_length)
    if args.sequences_per_label is None:
        real_labels = real.groupby("SequenceID")[TARGET].first().astype(int)
        target_counts = proportional_label_counts(real_labels.to_numpy(), args.total_sequences)
    else:
        target_counts = {label: args.sequences_per_label for label in range(-3, 4)}

    preview_session, preview_examples = sample_same_session_examples(
        examples[0], args.exemplars, args.seed, 0, 0
    )
    preview_prompt = make_prompt(
        args.mode,
        0,
        args.batch_size,
        preview_examples,
        session_stats[preview_session],
        global_stats,
        lower,
        upper,
        args.sequence_length,
    )
    preview_path = Path(args.preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(preview_prompt, encoding="utf-8")
    print(f"Prompt preview written to {preview_path}")

    if not args.execute:
        print("Dry run complete. Pass --execute to call DeepSeek.")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics)
    columns = ["SequenceID", "Step", *FEATURES, TARGET]
    rows: list[list] = []
    next_sequence_id = 0
    existing_counts = {label: 0 for label in range(-3, 4)}
    metrics = {
        "mode": args.mode,
        "model": "deepseek-chat",
        "temperature": args.temperature,
        "seed": args.seed,
        "sequence_length": args.sequence_length,
        "requested_sequences": int(sum(target_counts.values())),
        "target_counts": target_counts,
        "requests": [],
        "totals": {
            "attempts": 0,
            "returned_objects": 0,
            "accepted_sequences": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_seconds": 0.0,
            "rejections": {},
        },
    }
    if args.resume and output.exists():
        if metrics_path.exists():
            previous_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if previous_metrics.get("mode") != args.mode:
                raise ValueError("Resume metrics mode does not match --mode")
            metrics = previous_metrics
            metrics["requested_sequences"] = int(sum(target_counts.values()))
            metrics["target_counts"] = target_counts
        existing = pd.read_csv(output)
        missing_columns = set(columns) - set(existing.columns)
        if missing_columns:
            raise ValueError(f"Resume file is missing columns: {sorted(missing_columns)}")
        resume_lengths = existing.groupby("SequenceID").size()
        resume_labels = existing.groupby("SequenceID")[TARGET].nunique()
        if not (resume_lengths == args.sequence_length).all() or not (resume_labels == 1).all():
            raise ValueError("Resume file contains incomplete or mixed-label sequences")
        rows = existing[columns].values.tolist()
        next_sequence_id = int(existing["SequenceID"].max()) + 1 if len(existing) else 0
        labels = existing.groupby("SequenceID")[TARGET].first().astype(int)
        existing_counts.update(labels.value_counts().astype(int).to_dict())
        metrics["totals"]["accepted_sequences"] = int(len(labels))
        print(f"Resuming {len(labels)} validated sequences from {output}", flush=True)

    prior_elapsed = float(metrics["totals"].get("elapsed_seconds", 0.0))
    run_started = time.perf_counter()
    for label in range(-3, 4):
        generated = existing_counts[label]
        previous_attempts = [
            int(request["attempt"])
            for request in metrics.get("requests", [])
            if int(request["label"]) == label
        ]
        attempt = max(previous_attempts, default=-1) + 1
        target = target_counts[label]
        while generated < target and attempt < max(target * 2, 200):
            session_id, chosen = sample_same_session_examples(
                examples[label], args.exemplars, args.seed, label, attempt
            )
            count = min(args.batch_size, target - generated)
            prompt = make_prompt(
                args.mode,
                label,
                count,
                chosen,
                session_stats[session_id],
                global_stats,
                lower,
                upper,
                args.sequence_length,
            )
            started = time.perf_counter()
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=args.temperature,
                max_tokens=8_000,
            )
            elapsed = time.perf_counter() - started
            text = response.choices[0].message.content
            objects = extract_json(text)
            expanded, next_id, reasons = validate_direct(
                objects,
                args.mode,
                label,
                next_sequence_id,
                args.sequence_length,
                global_stats,
                lower,
                upper,
            )
            accepted = min(next_id - next_sequence_id, target - generated)
            if accepted < next_id - next_sequence_id:
                expanded = expanded[: accepted * args.sequence_length]
                next_id = next_sequence_id + accepted
            token_usage = usage_dict(response)
            exemplar_ids = [str(item["sequence_id"]) for item in chosen]
            metrics["requests"].append(
                {
                    "label": label,
                    "attempt": attempt,
                    "session_id": session_id,
                    "example_sequence_ids": exemplar_ids,
                    "requested": count,
                    "returned_objects": len(objects),
                    "accepted": accepted,
                    "elapsed_seconds": elapsed,
                    **token_usage,
                    "rejections": reasons,
                }
            )
            totals = metrics["totals"]
            totals["attempts"] += 1
            totals["returned_objects"] += len(objects)
            totals["accepted_sequences"] += accepted
            totals["elapsed_seconds"] = prior_elapsed + time.perf_counter() - run_started
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                totals[key] += token_usage[key]
            for reason, value in reasons.items():
                totals["rejections"][reason] = totals["rejections"].get(reason, 0) + value

            rows.extend(expanded)
            next_sequence_id = next_id
            generated += accepted
            attempt += 1
            pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
            save_metrics(metrics_path, metrics)
            print(
                f"mode={args.mode} label={label} accepted={generated}/{target} "
                f"attempt={attempt} returned={len(objects)} valid={accepted}",
                flush=True,
            )
            time.sleep(0.3)
        if generated < target:
            raise RuntimeError(
                f"Stopped after {attempt} attempts for label={label}: {generated}/{target}"
            )

    metrics["totals"]["elapsed_seconds"] = (
        prior_elapsed + time.perf_counter() - run_started
    )
    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
    save_metrics(metrics_path, metrics)
    print(f"Saved {next_sequence_id} sequences to {output}")
    print(f"Generation metrics written to {metrics_path}")


if __name__ == "__main__":
    main()
