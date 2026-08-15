"""Paper-aligned 2x2 utility evaluation for AutoTherm synthetic data.

The model and training target follow the authors' public implementation at
commit 0ad0df29d12b5de0333f7767a964908be382eee5. Real windows are constructed
inside numerically sorted sessions and sampled every 10 raw observations as
requested. Actual physical spans are measured from timestamps and saved with
the result instead of being inferred from an assumed frame rate.
Inputs use the four-feature combination reported best in paper Section 7.7.
"""

from __future__ import annotations

import argparse
import json
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

from temporal_utils import PAPER_BEST_FEATURES as FEATURES
from temporal_utils import TARGET, load_sessions


LABELS = np.arange(-3, 4)
NUM_CLASSES = len(LABELS)

# Fixed paper configuration. These are deliberately not CLI arguments.
SEEDS = (41, 42, 43, 44, 45)
SEQUENCE_LENGTH = 30
RAW_DOWNSAMPLE = 10
WINDOW_STRIDE = 1
HIDDEN_SIZE = 64
LSTM_LAYERS = 2
DROPOUT = 0.5
LEARNING_RATE = 1e-5
LEARNING_RATE_DECAY = 0.9999999
BATCH_SIZE = 16
MAX_EPOCHS = 100
REAL_VALIDATION_SESSIONS = 2

OFFICIAL_CODE_COMMIT = "0ad0df29d12b5de0333f7767a964908be382eee5"
FIXED_NORMALIZATION_BOUNDS = {
    "Ambient_Temperature": (15.0, 40.0),
    "Ambient_Humidity": (0.0, 100.0),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-dataset-root", default="F:/synthetic/dataset/indoor")
    parser.add_argument("--synthetic-train")
    parser.add_argument("--synthetic-test")
    parser.add_argument("--synthetic-flat")
    parser.add_argument("--method", required=True)
    parser.add_argument("--output-dir", default="F:/synthetic/results")
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Use a fixed optimizer-update budget instead of MAX_EPOCHS.",
    )
    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save one best-validation-accuracy model checkpoint per seed.",
    )
    parser.add_argument(
        "--synth-to-real-only",
        action="store_true",
        help="Run only Synthetic-to-Real; otherwise produce the full 2x2 matrix.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_indices(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=int)
    if not np.isin(values, LABELS).all():
        invalid = sorted(set(values.tolist()) - set(LABELS.tolist()))
        raise ValueError(f"Labels outside -3..3: {invalid}")
    return (values + 3).astype(np.int64)


def ordinal_targets(values: np.ndarray) -> np.ndarray:
    """Cheng-style cumulative targets used by the official repository."""
    indices = label_indices(values)
    positions = np.arange(NUM_CLASSES)[None, :]
    return (positions <= indices[:, None]).astype(np.float32)


def decode_ordinal(outputs: torch.Tensor) -> torch.Tensor:
    """Official-compatible decoding: round, count leading ordinal activations."""
    rounded = torch.round(outputs)
    return torch.clamp(rounded.sum(dim=1).long() - 1, 0, NUM_CLASSES - 1)


def validate_feature_frame(frame: pd.DataFrame, source: str) -> None:
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite feature values")
    label_indices(frame[TARGET].to_numpy(dtype=int))


def real_windows(
    dataset_root: str | Path, split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw HF records, numerically sort, downsample, and window per session."""
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_groups: list[str] = []
    all_spans: list[float] = []
    sessions = load_sessions(dataset_root, split)
    for session_id, frame in sessions.items():
        sampled = frame.iloc[::RAW_DOWNSAMPLE].reset_index(drop=True)
        values = sampled[FEATURES].to_numpy(dtype=np.float32)
        labels = sampled[TARGET].to_numpy(dtype=int)
        timestamps = sampled["Timestamp"].astype("int64").to_numpy() / 1e9
        for start in range(0, len(sampled) - SEQUENCE_LENGTH + 1, WINDOW_STRIDE):
            stop = start + SEQUENCE_LENGTH
            all_x.append(values[start:stop])
            all_y.append(int(labels[stop - 1]))
            all_groups.append(str(session_id))
            all_spans.append(float(timestamps[stop - 1] - timestamps[start]))
    if not all_x:
        raise ValueError(f"No real windows constructed for split={split}")
    return (
        np.asarray(all_x),
        np.asarray(all_y),
        np.asarray(all_groups),
        np.asarray(all_spans),
    )


def explicit_synthetic_windows(
    frame: pd.DataFrame, source: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_feature_frame(frame, source)
    required = {"SequenceID", "Step"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{source} must contain SequenceID and Step")
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_groups: list[str] = []
    for sequence_id, sequence in frame.groupby("SequenceID", sort=False):
        sequence = sequence.sort_values("Step")
        if len(sequence) != SEQUENCE_LENGTH:
            raise ValueError(
                f"{source} SequenceID={sequence_id} has {len(sequence)} rows, "
                f"expected {SEQUENCE_LENGTH}"
            )
        labels = sequence[TARGET].to_numpy(dtype=int)
        if not np.all(labels == labels[0]):
            raise ValueError(f"{source} SequenceID={sequence_id} contains mixed labels")
        all_x.append(sequence[FEATURES].to_numpy(dtype=np.float32))
        all_y.append(int(labels[0]))
        all_groups.append(str(sequence_id))
    if not all_x:
        raise ValueError(f"No explicit synthetic sequences found in {source}")
    return np.asarray(all_x), np.asarray(all_y), np.asarray(all_groups)


def flat_synthetic_windows(
    frame: pd.DataFrame, source: str, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Form fixed-label pseudo-sequences for row-wise SDV baselines."""
    validate_feature_frame(frame, source)
    rng = np.random.default_rng(seed)
    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_groups: list[str] = []
    group_number = 0
    for label in LABELS:
        subset = frame.loc[frame[TARGET].astype(int) == label]
        order = rng.permutation(len(subset))
        values = subset.iloc[order][FEATURES].to_numpy(dtype=np.float32)
        usable = (len(values) // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
        for start in range(0, usable, SEQUENCE_LENGTH):
            all_x.append(values[start : start + SEQUENCE_LENGTH])
            all_y.append(int(label))
            all_groups.append(f"flat-{group_number}")
            group_number += 1
    if not all_x:
        raise ValueError(f"No flat synthetic sequences found in {source}")
    return np.asarray(all_x), np.asarray(all_y), np.asarray(all_groups)


def split_real_train(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    unique_sessions = np.unique(groups)
    if len(unique_sessions) <= REAL_VALIDATION_SESSIONS:
        raise ValueError("Not enough real sessions for a held-out validation split")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=REAL_VALIDATION_SESSIONS, random_state=seed
    )
    train_index, validation_index = next(splitter.split(x, y, groups))
    return x[train_index], y[train_index], x[validation_index], y[validation_index]


def duplicate_aware_groups(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Assign exact duplicate sequences to one split to prevent leakage."""
    signature_owner: dict[tuple[int, bytes], str] = {}
    owners: list[str] = []
    for sequence, label, group in zip(x, y, groups):
        signature = (int(label), np.round(sequence, 7).tobytes())
        signature_owner.setdefault(signature, str(group))
        owners.append(signature_owner[signature])
    return np.asarray(owners)


def stratified_group_split(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int,
    include_test: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    owners = duplicate_aware_groups(x, y, groups)
    rng = np.random.default_rng(seed)
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    test_groups: set[str] = set()
    for label in LABELS:
        label_groups = np.unique(owners[y == label])
        if len(label_groups) < (3 if include_test else 2):
            raise ValueError(f"Label {label} has too few unique synthetic groups")
        rng.shuffle(label_groups)
        if include_test:
            n_test = max(1, int(round(len(label_groups) * 0.2)))
            n_validation = max(1, int(round(len(label_groups) * 0.2)))
            if n_test + n_validation >= len(label_groups):
                n_test = n_validation = 1
            test_groups.update(label_groups[:n_test].tolist())
            validation_groups.update(
                label_groups[n_test : n_test + n_validation].tolist()
            )
            train_groups.update(label_groups[n_test + n_validation :].tolist())
        else:
            n_validation = max(1, int(round(len(label_groups) * 0.2)))
            validation_groups.update(label_groups[:n_validation].tolist())
            train_groups.update(label_groups[n_validation:].tolist())

    train_mask = np.isin(owners, list(train_groups))
    validation_mask = np.isin(owners, list(validation_groups))
    if np.any(train_mask & validation_mask):
        raise RuntimeError("Synthetic train/validation groups overlap")
    test_x = test_y = None
    if include_test:
        test_mask = np.isin(owners, list(test_groups))
        if np.any((train_mask | validation_mask) & test_mask):
            raise RuntimeError("Synthetic test groups overlap training data")
        test_x, test_y = x[test_mask], y[test_mask]
    return (
        x[train_mask],
        y[train_mask],
        x[validation_mask],
        y[validation_mask],
        test_x,
        test_y,
    )


class PaperMinMaxScaler:
    """Official feature bounds with train-only bounds for Wrist and GSR."""

    def __init__(self) -> None:
        self.bounds: dict[str, tuple[float, float]] = {}

    def fit(self, values: np.ndarray) -> "PaperMinMaxScaler":
        flat = values.reshape(-1, len(FEATURES))
        for index, feature in enumerate(FEATURES):
            if feature in FIXED_NORMALIZATION_BOUNDS:
                lower, upper = FIXED_NORMALIZATION_BOUNDS[feature]
            else:
                lower = float(flat[:, index].min())
                upper = float(flat[:, index].max())
            if not upper > lower:
                raise ValueError(f"Degenerate normalization range for {feature}")
            self.bounds[feature] = (lower, upper)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.bounds:
            raise RuntimeError("Scaler must be fit before transform")
        result = values.astype(np.float32, copy=True)
        for index, feature in enumerate(FEATURES):
            lower, upper = self.bounds[feature]
            result[..., index] = (result[..., index] - lower) / (upper - lower)
        return result


class PaperLSTM(nn.Module):
    """Official AutoTherm LSTM encoder and sigmoid ordinal decoder."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            len(FEATURES),
            HIDDEN_SIZE,
            LSTM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE // 2)
        self.fc2 = nn.Linear(HIDDEN_SIZE // 2, NUM_CLASSES)
        self.activation = nn.Sigmoid()

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        sequence, _ = self.lstm(values)
        state = sequence[:, -1]
        state = self.fc1(state)
        state = self.dropout(state)
        return self.activation(self.fc2(state))


def score(
    model: nn.Module, x: np.ndarray, y: np.ndarray, device: torch.device
) -> dict[str, object]:
    model.eval()
    predictions: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=512, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            outputs = model(batch.to(device))
            predictions.append(decode_ordinal(outputs).cpu().numpy())
    predicted = np.concatenate(predictions)
    truth = label_indices(y)
    return {
        "acc": float(accuracy_score(truth, predicted)),
        "f1": float(
            f1_score(
                truth,
                predicted,
                labels=np.arange(NUM_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion": confusion_matrix(
            truth, predicted, labels=np.arange(NUM_CLASSES)
        ).tolist(),
    }


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    device: torch.device,
    seed: int,
    max_epochs: int = MAX_EPOCHS,
    max_updates: int | None = None,
) -> nn.Module:
    seed_everything(seed)
    model = PaperLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: LEARNING_RATE_DECAY**step
    )
    criterion = nn.MSELoss()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_train),
            torch.from_numpy(ordinal_targets(y_train)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    best_accuracy = -1.0
    best_state = None
    best_epoch = 0
    best_update = 0
    updates = 0
    epoch = 0
    while True:
        epoch += 1
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            updates += 1
            if max_updates is not None and updates >= max_updates:
                break
        validation = score(model, x_validation, y_validation, device)
        if float(validation["acc"]) > best_accuracy + 1e-12:
            best_accuracy = float(validation["acc"])
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            best_update = updates
        if epoch == 1 or epoch % 10 == 0:
            budget = (
                f"updates={updates}/{max_updates}"
                if max_updates is not None
                else f"epoch={epoch}/{max_epochs}"
            )
            print(
                f"    {budget} "
                f"val_acc={validation['acc']:.4f} val_f1={validation['f1']:.4f}",
                flush=True,
            )
        if max_updates is not None:
            if updates >= max_updates:
                break
        elif epoch >= max_epochs:
            break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.training_metadata = {
        "completed_updates": updates,
        "completed_epochs": epoch,
        "best_epoch": best_epoch,
        "best_update": best_update,
        "best_validation_accuracy": best_accuracy,
    }
    print(
        f"    restored epoch={best_epoch} update={best_update} "
        f"val_acc={best_accuracy:.4f}",
        flush=True,
    )
    return model


def load_synthetic_pool(
    path: str, force_flat: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if force_flat:
        return flat_synthetic_windows(frame, path)
    return explicit_synthetic_windows(frame, path)


def span_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def protocol_metadata(
    real_train_spans: np.ndarray, real_test_spans: np.ndarray
) -> dict[str, object]:
    return {
        "official_code_commit": OFFICIAL_CODE_COMMIT,
        "features": FEATURES,
        "labels": LABELS.tolist(),
        "sequence_length": SEQUENCE_LENGTH,
        "raw_downsample": RAW_DOWNSAMPLE,
        "physical_span_seconds_train": span_summary(real_train_spans),
        "physical_span_seconds_test": span_summary(real_test_spans),
        "window_stride": WINDOW_STRIDE,
        "hidden_size": HIDDEN_SIZE,
        "lstm_layers": LSTM_LAYERS,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "learning_rate_decay_per_step": LEARNING_RATE_DECAY,
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "loss": "MSE on cumulative ordinal targets",
        "optimizer": "Adam",
        "seeds": list(SEEDS),
        "normalization_fixed_bounds": FIXED_NORMALIZATION_BOUNDS,
        "normalization_train_fitted": [
            "Wrist_Skin_Temperature",
            "GSR",
        ],
    }


def main() -> None:
    args = parse_args()
    if bool(args.synthetic_flat) == bool(args.synthetic_train):
        raise ValueError("Provide exactly one of --synthetic-flat or --synthetic-train")
    if args.synthetic_flat and args.synthetic_test:
        raise ValueError("--synthetic-test is only supported with --synthetic-train")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(
        character if character.isalnum() else "_" for character in args.method.lower()
    ).strip("_")
    print("Loading and windowing numerically sorted real sessions...", flush=True)
    real_x, real_y, real_groups, real_train_spans = real_windows(
        args.real_dataset_root, "train"
    )
    real_test_x, real_test_y, _, real_test_spans = real_windows(
        args.real_dataset_root, "test"
    )
    print(f"Real windows: train_pool={len(real_x)} test={len(real_test_x)}", flush=True)
    print(
        "Observed first-to-last span (s): "
        f"train_median={np.median(real_train_spans):.3f} "
        f"test_median={np.median(real_test_spans):.3f}",
        flush=True,
    )

    synthetic_path = args.synthetic_flat or args.synthetic_train
    synthetic_x, synthetic_y, synthetic_groups = load_synthetic_pool(
        synthetic_path, force_flat=bool(args.synthetic_flat)
    )
    external_synthetic_test = None
    if args.synthetic_test:
        external_synthetic_test = load_synthetic_pool(args.synthetic_test, force_flat=False)
    print(f"Synthetic sequence pool: {len(synthetic_x)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_rows: list[dict[str, object]] = []
    confusion: dict[str, list[list[int]]] = {}
    for seed in SEEDS:
        print(f"seed={seed}", flush=True)
        rtr_x, rtr_y, rval_x, rval_y = split_real_train(
            real_x, real_y, real_groups, seed
        )

        include_internal_test = not args.synth_to_real_only and external_synthetic_test is None
        str_x, str_y, sval_x, sval_y, ste_x, ste_y = stratified_group_split(
            synthetic_x,
            synthetic_y,
            synthetic_groups,
            seed,
            include_test=include_internal_test,
        )
        if external_synthetic_test is not None:
            ste_x, ste_y, _ = external_synthetic_test

        scaler = PaperMinMaxScaler().fit(rtr_x)
        rtr_x = scaler.transform(rtr_x)
        rval_x = scaler.transform(rval_x)
        rte_x = scaler.transform(real_test_x)
        str_x = scaler.transform(str_x)
        sval_x = scaler.transform(sval_x)
        if ste_x is not None:
            ste_x = scaler.transform(ste_x)

        real_model = None
        if not args.synth_to_real_only:
            print("  training Real model", flush=True)
            real_model = train_model(
                rtr_x,
                rtr_y,
                rval_x,
                rval_y,
                device,
                seed,
                max_updates=args.max_updates,
            )
        print("  training Synthetic model", flush=True)
        synthetic_model = train_model(
            str_x,
            str_y,
            sval_x,
            sval_y,
            device,
            seed,
            max_updates=args.max_updates,
        )

        if args.save_checkpoints:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{safe_name}_seed{seed}.pt"
            torch.save(
                {
                    "method": args.method,
                    "seed": seed,
                    "model_state_dict": synthetic_model.state_dict(),
                    "training_metadata": synthetic_model.training_metadata,
                    "scaler_bounds": scaler.bounds,
                    "features": FEATURES,
                    "labels": LABELS.tolist(),
                    "sequence_length": SEQUENCE_LENGTH,
                    "raw_downsample": RAW_DOWNSAMPLE,
                    "learning_rate": LEARNING_RATE,
                    "learning_rate_decay_per_step": LEARNING_RATE_DECAY,
                    "batch_size": BATCH_SIZE,
                    "max_updates": args.max_updates,
                },
                checkpoint_path,
            )
            print(f"  saved checkpoint: {checkpoint_path}", flush=True)

        directions = {
            f"{args.method}->Real": score(
                synthetic_model, rte_x, real_test_y, device
            )
        }
        if not args.synth_to_real_only:
            if ste_x is None or ste_y is None or real_model is None:
                raise RuntimeError("Full 2x2 evaluation requires a synthetic test split")
            directions.update(
                {
                    "Real->Real": score(real_model, rte_x, real_test_y, device),
                    f"Real->{args.method}": score(real_model, ste_x, ste_y, device),
                    f"{args.method}->{args.method}": score(
                        synthetic_model, ste_x, ste_y, device
                    ),
                }
            )
        for direction, metrics in directions.items():
            print(
                f"  {direction}: F1={metrics['f1']:.4f} Acc={metrics['acc']:.4f}",
                flush=True,
            )
            result_rows.append(
                {
                    "method": args.method,
                    "seed": seed,
                    "direction": direction,
                    "f1": metrics["f1"],
                    "acc": metrics["acc"],
                }
            )
            confusion[f"{seed}:{direction}"] = metrics["confusion"]

    results = pd.DataFrame(result_rows)
    raw_path = output_dir / f"2x2_paper_lstm_{safe_name}_raw.csv"
    summary_path = output_dir / f"2x2_paper_lstm_{safe_name}_summary.csv"
    confusion_path = output_dir / f"2x2_paper_lstm_{safe_name}_confusion.json"
    protocol_path = output_dir / f"2x2_paper_lstm_{safe_name}_protocol.json"
    results.to_csv(raw_path, index=False)
    summary = (
        results.groupby(["method", "direction"])[["f1", "acc"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    confusion_path.write_text(json.dumps(confusion, indent=2), encoding="utf-8")
    protocol = protocol_metadata(real_train_spans, real_test_spans)
    protocol.update(
        {
            "training_budget_mode": (
                "fixed_optimizer_updates" if args.max_updates is not None else "fixed_epochs"
            ),
            "max_epochs": None if args.max_updates is not None else MAX_EPOCHS,
            "max_optimizer_updates": args.max_updates,
            "checkpoint_selection": "validation_accuracy",
        }
    )
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
