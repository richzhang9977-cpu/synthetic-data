from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_from_disk


FEATURES = [
    "Radiation-Temp",
    "Wrist_Skin_Temperature",
    "GSR",
    "Ambient_Temperature",
    "Ambient_Humidity",
]
PAPER_BEST_FEATURES = [
    "Wrist_Skin_Temperature",
    "GSR",
    "Ambient_Temperature",
    "Ambient_Humidity",
]
TARGET = "Label"
META_COLUMNS = ["file_name", "Timestamp"]
FRAME_RE = re.compile(r"frame_f_(\d+)")


def frame_number(file_name: str) -> int:
    """Extract the numeric video-frame index from a dataset file name."""
    match = FRAME_RE.search(str(file_name))
    if not match:
        raise ValueError(f"Cannot extract frame number from {file_name!r}")
    return int(match.group(1))


def load_sessions(dataset_root: str | Path, split: str) -> dict[str, pd.DataFrame]:
    """Load one HF split into numerically ordered, session-level DataFrames."""
    dataset_path = Path(dataset_root) / split
    columns = META_COLUMNS + FEATURES + [TARGET]
    dataset = load_from_disk(str(dataset_path)).select_columns(columns)
    buffers: dict[str, dict[str, list]] = defaultdict(
        lambda: {"frame": [], **{column: [] for column in columns if column != "file_name"}}
    )

    for batch in dataset.iter(batch_size=50_000):
        for index, file_name in enumerate(batch["file_name"]):
            session_id = str(file_name).split("/")[0]
            buffer = buffers[session_id]
            buffer["frame"].append(frame_number(file_name))
            for column in columns:
                if column != "file_name":
                    buffer[column].append(batch[column][index])

    sessions: dict[str, pd.DataFrame] = {}
    for session_id, buffer in buffers.items():
        frame = pd.DataFrame(buffer)
        frame["Timestamp"] = pd.to_datetime(frame["Timestamp"], errors="raise")
        for feature in FEATURES:
            frame[feature] = pd.to_numeric(frame[feature], errors="raise").astype(float)
        frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
        frame = frame.sort_values("frame", kind="stable").reset_index(drop=True)
        if frame["frame"].duplicated().any():
            raise ValueError(f"Duplicate frame numbers in session {session_id}")
        sessions[session_id] = frame
    return sessions


def exact_run_lengths(values: np.ndarray) -> np.ndarray:
    """Lengths of consecutive exactly-equal runs."""
    values = np.asarray(values)
    if not len(values):
        return np.array([], dtype=int)
    boundaries = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
    return np.diff(boundaries)


def session_deltas(sessions: dict[str, pd.DataFrame], column: str) -> np.ndarray:
    """Concatenate first differences without crossing session boundaries."""
    return np.concatenate(
        [np.diff(frame[column].to_numpy(dtype=float)) for frame in sessions.values() if len(frame) > 1]
    )


def proportional_label_counts(values, total: int) -> dict[int, int]:
    """Allocate an exact total across labels -3..3 using largest remainders."""
    counts = pd.Series(values, dtype=int).value_counts().reindex(range(-3, 4), fill_value=0)
    if (counts == 0).any():
        missing = counts.index[counts == 0].tolist()
        raise ValueError(f"Cannot allocate labels absent from training data: {missing}")
    exact = counts.to_numpy(dtype=float) * (total / counts.sum())
    allocated = np.floor(exact).astype(int)
    remainder_order = np.argsort(-(exact - allocated), kind="stable")
    for index in remainder_order[: total - int(allocated.sum())]:
        allocated[index] += 1
    return {int(label): int(value) for label, value in zip(counts.index, allocated)}
