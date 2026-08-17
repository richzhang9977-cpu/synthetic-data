from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_utils import PAPER_BEST_FEATURES as FEATURES
from temporal_utils import TARGET, proportional_label_counts


SHORT = {
    "Wrist_Skin_Temperature": "wrist",
    "GSR": "gsr",
    "Ambient_Temperature": "ambient",
    "Ambient_Humidity": "humidity",
}
LONG = {short: long for long, short in SHORT.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--real-train",
        default="F:/synthetic/dataset/processed_every10/train_sequences.csv",
    )
    parser.add_argument(
        "--output",
        help="Output CSV; defaults to a mode-specific every10 path.",
    )
    parser.add_argument(
        "--preview",
        help="Prompt preview; defaults to a mode-specific every10 path.",
    )
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument("--total-sequences", type=int, default=1_680)
    parser.add_argument(
        "--sequences-per-label",
        type=int,
        help="Optional balanced-generation override for an ablation run.",
    )
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--exemplars", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--prompt-mode",
        choices=("v5-controlled", "sparse-naive"),
        default="v5-controlled",
        help=(
            "v5-controlled includes session dynamics and global statistics; "
            "sparse-naive uses only event-form same-session examples."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from --output and top up every label to --sequences-per-label.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output is None:
        filename = (
            "approach_a_v5_events_b10.csv"
            if args.prompt_mode == "v5-controlled"
            else "sparse_naive_events_b10.csv"
        )
        args.output = f"F:/synthetic/llm_synthetic/every10/{filename}"
    if args.preview is None:
        filename = (
            "approach_a_v5_b10_prompt_preview.txt"
            if args.prompt_mode == "v5-controlled"
            else "sparse_naive_b10_prompt_preview.txt"
        )
        args.preview = f"F:/synthetic/results/every10/{filename}"
    return args


def empirical_stats(real: pd.DataFrame) -> tuple[dict, dict, dict]:
    stats = {}
    for feature in FEATURES:
        delta = np.concatenate(
            [
                np.diff(group.sort_values("Step")[feature].to_numpy(dtype=float))
                for _, group in real.groupby("SequenceID", sort=False)
            ]
        )
        absolute = np.abs(delta)
        nonzero = absolute[absolute > 1e-12]
        stats[SHORT[feature]] = {
            "change_probability_per_step": float((absolute > 1e-12).mean()),
            "median_nonzero_abs_delta": float(np.median(nonzero)),
            "p95_nonzero_abs_delta": float(np.percentile(nonzero, 95)),
            "max_abs_delta": float(absolute.max()),
        }
    lower = real[FEATURES].quantile(0.001).to_dict()
    upper = real[FEATURES].quantile(0.999).to_dict()
    return stats, lower, upper


def session_dynamics_stats(real: pd.DataFrame, global_stats: dict) -> dict[str, dict]:
    session_stats = {}
    for session_id, session in real.groupby("session_id", sort=False):
        feature_stats = {}
        for feature in FEATURES:
            delta = np.concatenate(
                [
                    np.diff(sequence.sort_values("Step")[feature].to_numpy(dtype=float))
                    for _, sequence in session.groupby("SequenceID", sort=False)
                ]
            )
            absolute = np.abs(delta)
            nonzero = absolute[absolute > 1e-12]
            short_name = SHORT[feature]
            feature_stats[short_name] = {
                "change_probability_per_step": float((absolute > 1e-12).mean()),
                "median_nonzero_abs_delta": (
                    float(np.median(nonzero))
                    if len(nonzero)
                    else global_stats[short_name]["median_nonzero_abs_delta"]
                ),
                "p95_nonzero_abs_delta": (
                    float(np.percentile(nonzero, 95))
                    if len(nonzero)
                    else global_stats[short_name]["p95_nonzero_abs_delta"]
                ),
            }
        session_stats[str(session_id)] = feature_stats
    return session_stats


def event_summary(window: pd.DataFrame) -> dict:
    values = window[FEATURES].to_numpy(dtype=float)
    updates = []
    for step, delta in enumerate(np.diff(values, axis=0), start=1):
        changed = {
            SHORT[feature]: round(float(delta[index]), 7)
            for index, feature in enumerate(FEATURES)
            if abs(delta[index]) > 1e-12
        }
        if changed:
            updates.append({"step": step, "deltas": changed})
    return {
        "label": int(window[TARGET].iloc[0]),
        "base": {SHORT[feature]: round(float(window[feature].iloc[0]), 7) for feature in FEATURES},
        "updates": updates,
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
        examples[label].setdefault(session_id, []).append(event_summary(window))
    return examples


def sample_same_session_examples(
    candidates_by_session: dict[str, list[dict]],
    count: int,
    rng: np.random.Generator,
) -> tuple[str, list[dict]]:
    eligible = [
        (session_id, session_examples)
        for session_id, session_examples in candidates_by_session.items()
        if len(session_examples) >= count
    ]
    if not eligible:
        largest = max((len(items) for items in candidates_by_session.values()), default=0)
        raise ValueError(
            f"No single session contains {count} eligible examples; largest has {largest}"
        )
    session_id, session_examples = eligible[int(rng.integers(len(eligible)))]
    selected = rng.choice(len(session_examples), count, replace=False)
    return session_id, [session_examples[index] for index in selected]


def make_prompt(
    prompt_mode: str,
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
    global_max_delta = {
        SHORT[feature]: global_stats[SHORT[feature]]["max_abs_delta"]
        for feature in FEATURES
    }
    common = f"""Generate {count} synthetic thermal-comfort sensor sequences as a JSON array.

Time scale and semantics:
- Each sequence contains {length} sampled observations.
- Consecutive observations are every 10th numerically ordered raw record.
- A {length}-step sequence covers approximately 10.7 seconds; timestamps vary slightly.
- Every sequence has fixed label={label}; do not generate label transitions.
- Sensor readings are piecewise constant. Updates are sparse events, not smooth interpolation.
- Generate a correlated four-feature base state, then only the updates.
"""

    schema = f"""Schema for each array element:
{{"label": {label}, "base": {{"wrist": number, "gsr": number,
"ambient": number, "humidity": number}}, "updates": [
{{"step": integer from 1 to {length - 1}, "deltas": {{"gsr": number}}}}
]}}"""

    if prompt_mode == "sparse-naive":
        return common + f"""
Rules:
- Preserve realistic dependence among base wrist, gsr, ambient, and humidity.
- Return JSON only; no Markdown or explanation.

{schema}

Use only the following same-participant-session, same-label training examples
to infer the value scale, cross-feature relationships, event frequency, and
event magnitudes.

Training-only examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}
"""

    if prompt_mode != "v5-controlled":
        raise ValueError(f"Unsupported prompt mode: {prompt_mode}")

    return common + f"""
Current-session dynamics:
{json.dumps(session_stats, ensure_ascii=False, indent=2)}

Global safety limits - feature bounds:
{json.dumps(bounds, ensure_ascii=False)}

Global safety limits - maximum allowed change per update:
{json.dumps(global_max_delta, ensure_ascii=False, indent=2)}

Rules:
- Never exceed the supplied feature bounds.
- Never exceed the supplied global maximum allowed change for an individual update.
- Ambient updates are normally 0.1 C and at most 0.3 C.
- Humidity updates are integer steps of 1, rarely 2.
- Preserve realistic dependence among base wrist, gsr, ambient, and humidity.
- Return JSON only; no Markdown or explanation.

{schema}

Training-only examples:
{json.dumps(examples, ensure_ascii=False, indent=2)}
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
            # Long generations are occasionally truncated or malformed. Treat
            # those responses as rejected candidates instead of losing all
            # accepted sequences from the run.
            return []
    return value if isinstance(value, list) else []


def validate_and_expand(
    objects: list,
    target_label: int,
    start_sequence_id: int,
    length: int,
    stats: dict,
    lower: dict,
    upper: dict,
) -> tuple[list[list], int]:
    rows = []
    sequence_id = start_sequence_id
    for obj in objects:
        try:
            if int(obj["label"]) != target_label:
                continue
            state = np.array([float(obj["base"][SHORT[feature]]) for feature in FEATURES])
            lo = np.array([lower[feature] for feature in FEATURES], dtype=float)
            hi = np.array([upper[feature] for feature in FEATURES], dtype=float)
            if np.any(state < lo) or np.any(state > hi):
                continue
            updates = {}
            valid = True
            for update in obj.get("updates", []):
                step = int(update["step"])
                if not 1 <= step < length:
                    valid = False
                    break
                deltas = np.zeros(len(FEATURES), dtype=float)
                for short_name, value in update.get("deltas", {}).items():
                    if short_name not in LONG:
                        valid = False
                        break
                    feature = LONG[short_name]
                    index = FEATURES.index(feature)
                    delta = float(value)
                    if abs(delta) > stats[short_name]["max_abs_delta"] + 1e-9:
                        valid = False
                        break
                    deltas[index] = delta
                if not valid:
                    break
                updates[step] = updates.get(step, np.zeros(len(FEATURES))) + deltas
            if not valid:
                continue
            sequence = []
            current = state.copy()
            for step in range(length):
                if step in updates:
                    current = current + updates[step]
                if np.any(current < lo) or np.any(current > hi):
                    valid = False
                    break
                sequence.append(current.copy())
            if not valid:
                continue
            for step, values in enumerate(sequence):
                rows.append([sequence_id, step, *values.tolist(), target_label])
            sequence_id += 1
        except (KeyError, TypeError, ValueError):
            continue
    return rows, sequence_id


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    real = pd.read_csv(args.real_train)
    required = {"SequenceID", "Step", "session_id", *FEATURES, TARGET}
    missing = required - set(real.columns)
    if missing:
        raise ValueError(f"Shared sequence input is missing columns: {sorted(missing)}")
    lengths = real.groupby("SequenceID").size()
    mixed = real.groupby("SequenceID")[TARGET].nunique()
    if not (lengths == args.sequence_length).all() or not (mixed == 1).all():
        raise ValueError("Shared sequence input must contain fixed-label 30-step sequences")
    stats, lower, upper = empirical_stats(real)
    session_stats = session_dynamics_stats(real, stats)
    examples = collect_examples(real, args.sequence_length)
    if args.sequences_per_label is None:
        real_sequence_labels = real.groupby("SequenceID")[TARGET].first().astype(int)
        target_counts = proportional_label_counts(
            real_sequence_labels.to_numpy(), args.total_sequences
        )
    else:
        target_counts = {label: args.sequences_per_label for label in range(-3, 4)}

    preview_session_id, preview_examples = sample_same_session_examples(
        examples[0], args.exemplars, rng
    )
    preview_prompt = make_prompt(
        args.prompt_mode,
        0,
        args.batch_size,
        preview_examples,
        session_stats[preview_session_id],
        stats,
        lower,
        upper,
        args.sequence_length,
    )
    preview_path = Path(args.preview)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    preview_path.write_text(preview_prompt, encoding="utf-8")
    print(f"Prompt preview written to {preview_path}")

    if not args.execute:
        print("Dry run complete. Pass --execute after rotating DEEPSEEK_API_KEY.")
        return

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["SequenceID", "Step", *FEATURES, TARGET]
    rows = []
    next_sequence_id = 0
    existing_counts = {label: 0 for label in range(-3, 4)}
    if args.resume and output.exists():
        existing = pd.read_csv(output)
        missing_columns = set(columns) - set(existing.columns)
        if missing_columns:
            raise ValueError(f"Resume file is missing columns: {sorted(missing_columns)}")
        lengths = existing.groupby("SequenceID").size()
        label_counts = existing.groupby("SequenceID")[TARGET].nunique()
        if not (lengths == args.sequence_length).all() or not (label_counts == 1).all():
            raise ValueError("Resume file contains incomplete or mixed-label sequences")
        rows = existing[columns].values.tolist()
        next_sequence_id = int(existing["SequenceID"].max()) + 1 if len(existing) else 0
        sequence_labels = existing.groupby("SequenceID")[TARGET].first()
        existing_counts.update(sequence_labels.value_counts().astype(int).to_dict())
        print(
            f"Resuming {len(sequence_labels)} validated sequences from {output}",
            flush=True,
        )
    for label in range(-3, 4):
        generated = existing_counts[label]
        attempts = 0
        target = target_counts[label]
        while generated < target and attempts < max(target, 100):
            session_id, chosen = sample_same_session_examples(
                examples[label], args.exemplars, rng
            )
            count = min(args.batch_size, target - generated)
            prompt = make_prompt(
                args.prompt_mode,
                label,
                count,
                chosen,
                session_stats[session_id],
                stats,
                lower,
                upper,
                args.sequence_length,
            )
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=8_000,
            )
            objects = extract_json(response.choices[0].message.content)
            expanded, next_id = validate_and_expand(
                objects,
                label,
                next_sequence_id,
                args.sequence_length,
                stats,
                lower,
                upper,
            )
            accepted = next_id - next_sequence_id
            rows.extend(expanded)
            next_sequence_id = next_id
            generated += accepted
            attempts += 1
            pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
            print(
                f"label={label} accepted={generated}/{target} "
                f"attempt={attempts}",
                flush=True,
            )
            time.sleep(0.3)

    pd.DataFrame(rows, columns=columns).to_csv(output, index=False)
    print(f"Saved {next_sequence_id} sequences to {output}")


if __name__ == "__main__":
    main()
