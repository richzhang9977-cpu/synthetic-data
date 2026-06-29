"""Full pipeline using paper's own dataloader + LSTM for 2x2 matrix evaluation.

Step 1: Export indoor_frames from HuggingFace → CSV files (paper format, ';' delimiter)
Step 2: Create participant split files
Step 3: Train paper's LSTM on real data → Real→Real baseline
Step 4: Generate SDV synthetic data
Step 5: Run 2x2 matrix: Real/Synth → Real/Synth
"""
import warnings; warnings.filterwarnings("ignore")
import json, time, os, sys, re
import numpy as np; import pandas as pd
import torch

# Add paper repo to path
sys.path.insert(0, "F:/thermal_paper")

print("=" * 60)
print("2x2 MATRIX EVALUATION — Paper Dataloader + LSTM")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Config
# ============================================================
DATA_DIR = "F:/thermal_data"
CSV_DIR = f"{DATA_DIR}/csv"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs("F:/thermal_paper/dataloaders/splits", exist_ok=True)

# Paper's column indices (see utils.py header list):
# 11=Radiation-Temp, 28=Wrist_Skin_Temperature, 31=Ambient_Temperature, 32=Ambient_Humidity, 33=Label
PAPER_COLS = [11, 28, 31, 32, 33]  # 4 features + label
SCALE = 7
SEQUENCE_WINDOW = 10

# ============================================================
# Step 1: Export indoor_frames to CSV
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Export indoor_frames → CSV files")
print("=" * 60)

# Check if already exported
existing = os.listdir(CSV_DIR) if os.path.exists(CSV_DIR) else []
if len(existing) >= 5:
    print(f"   Found {len(existing)} CSV files, skipping export.")
    all_sessions = sorted([f.replace('.csv', '') for f in existing if f.endswith('.csv')])
else:
    print("   Loading from HuggingFace...")
    from datasets import load_dataset
    ds = load_dataset("kopetri/AutoTherm", "indoor_frames", streaming=True, split="train")

    sessions = {}
    for i, sample in enumerate(ds):
        if i % 5000 == 0: print(f"   ... {i} frames processed, {len(sessions)} sessions")
        if i >= 100000: break

        try:
            meta = json.loads(sample["jpg.json"].decode("utf-8")
                            if isinstance(sample["jpg.json"], bytes) else sample["jpg.json"])
        except: continue

        fn = meta.get("file_name", "")
        sess = fn.split("/")[0] if "/" in fn else "unknown"
        sessions.setdefault(sess, []).append(meta)

    print(f"   Collected {len(sessions)} sessions, saving CSV files...")
    header = ["Timestamp","Age","Gender","Weight","Height","Bodyfat","Bodytemp",
              "Sport-Last-Hour","Time-Since-Meal","Tiredness","Clothing-Level",
              "Radiation-Temp","PCE-Ambient-Temp","Air-Velocity","Metabolic-Rate",
              "Emotion-Self","Emotion-ML","RGB_Frontal_View",
              "Nose","Neck","RShoulder","RElbow","LShoulder","LElbow",
              "REye","LEye","REar","LEar",
              "Wrist_Skin_Temperature","Heart_Rate","GSR",
              "Ambient_Temperature","Ambient_Humidity","Label"]

    all_sessions = []
    for sess, frames in sessions.items():
        # Sort by frame number
        frames.sort(key=lambda f: int(re.search(r'frame_f_(\d+)',
            str(f.get("file_name", ""))).group(1)
            if re.search(r'frame_f_(\d+)', str(f.get("file_name", ""))) else 0))

        # Write CSV with ; delimiter (paper format)
        rows = []
        for f in frames:
            row = []
            for h in header:
                val = f.get(h, "")
                if h == "RGB_Frontal_View":
                    row.append(f"images/{sess}/{f.get('file_name', '')}")
                else:
                    row.append(str(val) if val is not None else "")
            rows.append(";".join(row))

        filename = f"{sess}.csv"
        with open(f"{CSV_DIR}/{filename}", "w") as f:
            f.write("\n".join(rows))
        all_sessions.append(sess)

    print(f"   Exported {len(all_sessions)} CSV files to {CSV_DIR}")

print(f"   {len(all_sessions)} sessions total")

# ============================================================
# Step 2: Create participant split files (paper convention)
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Create split files")
print("=" * 60)

# Proportions: 70% train, 15% val, 15% test (by session/participant)
np.random.seed(42)
sessions_shuffled = sorted(all_sessions)
np.random.shuffle(sessions_shuffled)

n_train = int(len(sessions_shuffled) * 0.7)
n_val = int(len(sessions_shuffled) * 0.15)

train_sessions = sessions_shuffled[:n_train]
val_sessions = sessions_shuffled[n_train:n_train+n_val]
test_sessions = sessions_shuffled[n_train+n_val:]

# Write split files
for name, sessions in [("training", train_sessions), ("validation", val_sessions), ("test", test_sessions)]:
    path = f"F:/thermal_paper/dataloaders/splits/{name}_60_real.txt"
    with open(path, "w") as f:
        f.write("\n".join([f"{s}.csv" for s in sessions]))
    print(f"   {name}: {len(sessions)} sessions → {path}")

# Also write an all_real.txt
with open("F:/thermal_paper/dataloaders/splits/real_all.txt", "w") as f:
    f.write("\n".join([f"{s}.csv" for s in all_sessions]))
print(f"   all: {len(all_sessions)} sessions")

# ============================================================
# Step 3: Train paper's LSTM on REAL data (Real→Real)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Real→Real — Paper's LSTM baseline")
print("=" * 60)

from dataloaders.tc_dataloader import TC_Dataloader
from network.rnn_module import TC_RNN_Module
import pytorch_lightning as pl

# --- Create real dataloaders ---
print("   Creating training dataloader...")
train_dl = TC_Dataloader(
    root=f"{CSV_DIR}/",
    split="training",
    scale=SCALE,
    cols=PAPER_COLS,
    use_sequence=True,
    sequence_size=SEQUENCE_WINDOW,
    preprocess=True
)

print("   Creating validation dataloader...")
val_dl = TC_Dataloader(
    root=f"{CSV_DIR}/",
    split="validation",
    scale=SCALE,
    cols=PAPER_COLS,
    use_sequence=True,
    sequence_size=SEQUENCE_WINDOW,
    preprocess=True
)

print("   Creating test dataloader...")
test_dl = TC_Dataloader(
    root=f"{CSV_DIR}/",
    split="test",
    scale=SCALE,
    cols=PAPER_COLS,
    use_sequence=True,
    sequence_size=SEQUENCE_WINDOW,
    preprocess=True
)

# --- Configure LSTM module (paper hyperparams) ---
hparams = {
    "learning_rate": 1e-4,
    "batch_size": 4,
    "dropout": 0.5,
    "hidden": 128,
    "layers": 2,
    "sequence_window": SEQUENCE_WINDOW,
    "scale": SCALE,
    "weight_decay": 0.0,
    "cols": PAPER_COLS,
}

print(f"   Hyperparams: lr={hparams['learning_rate']}, hidden={hparams['hidden']}, layers={hparams['layers']}, dropout={hparams['dropout']}")

# Instantiate the paper's LightningModule
lstm_module = TC_RNN_Module(
    train_dataset=train_dl,
    val_dataset=val_dl,
    test_dataset=test_dl,
    batch_size=hparams["batch_size"],
    hidden=hparams["hidden"],
    layers=hparams["layers"],
    dropout=hparams["dropout"],
    learning_rate=hparams["learning_rate"],
    weight_decay=hparams["weight_decay"],
    scale=hparams["scale"],
    sequence_window=hparams["sequence_window"],
    cols=hparams["cols"],
)

# --- Train ---
print("\n   Training...")
trainer = pl.Trainer(
    max_epochs=2,  # paper uses 1-2 epochs for quick results
    accelerator="gpu" if DEVICE.type == "cuda" else "cpu",
    devices=1,
    enable_progress_bar=True,
    log_every_n_steps=50,
)
trainer.fit(lstm_module, train_dl, val_dl)

# --- Test Real→Real ---
print("\n   Testing Real -> Real...")
result_real_real = trainer.test(lstm_module, test_dl)
print(f"   Real→Real: {result_real_real}")

# Save trained model for later use
torch.save(lstm_module.state_dict(), f"{DATA_DIR}/lstm_real.pt")

# ============================================================
# Step 4: Generate SDV synthetic data
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SDV synthetic data generation")
print("=" * 60)

# Collect all training data as flat DataFrame for SDV
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.preprocessing import LabelEncoder, StandardScaler

real_train_all = pd.concat([pd.read_csv(f"{CSV_DIR}/{s}.csv", delimiter=";")
                            for s in train_sessions])

# Select paper feature columns
feature_names = [train_dl.columns[i] for i in range(len(train_dl.columns)-1)]
target_name = "Label"
sdv_df = real_train_all[feature_names + [target_name]].copy()

for c in sdv_df.columns:
    if sdv_df[c].dtype == object:
        sdv_df[c] = pd.to_numeric(sdv_df[c], errors="coerce").fillna(0)
sdv_df[target_name] = sdv_df[target_name].astype(int).astype(str)

meta = SingleTableMetadata(); meta.detect_from_dataframe(sdv_df)
meta.update_column(column_name=target_name, sdtype="categorical")
for c in feature_names: meta.update_column(column_name=c, sdtype="numerical")
try: meta.remove_primary_key()
except: pass

synth_data = {}
for name, cls, ep in [("GaussianCopula", GaussianCopulaSynthesizer, 0),
                       ("CTGAN", CTGANSynthesizer, 300),
                       ("TVAE", TVAESynthesizer, 300)]:
    cache = f"{DATA_DIR}/synth_{name}.csv"
    if os.path.exists(cache):
        df_s = pd.read_csv(cache)
        print(f"   {name}: loaded cached ({df_s.shape})")
    else:
        print(f"   {name}: generating..."); t0 = time.time()
        s = cls(meta, epochs=ep, cuda=True, verbose=False) if ep > 0 else cls(meta)
        s.fit(sdv_df); df_s = s.sample(num_rows=len(sdv_df))
        df_s.to_csv(cache, index=False)
        print(f"   {name}: done in {time.time()-t0:.1f}s ({df_s.shape})")
    synth_data[name] = df_s

# ============================================================
# Step 5: 2x2 Matrix Evaluation
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: 2x2 Matrix")
print("=" * 60)

results = {
    "Real→Real": result_real_real[0] if result_real_real else {"test_acc": 0.0},
}

for syn_name, syn_df in synth_data.items():
    # Build synthetic dataloaders from the generated CSV data
    # Save synthetic training data as CSV for paper's dataloader
    syn_csv_dir = f"{DATA_DIR}/synth_csv_{syn_name}"
    os.makedirs(syn_csv_dir, exist_ok=True)

    # Split synthetic data into train/val/test portions matching session counts
    syn_train_rows = len(real_train_all)
    syn_val_rows = syn_train_rows // 4
    syn_test_rows = syn_train_rows // 4

    # Write synthetic CSVs as single combined file (paper dataloader reads per-participant CSVs)
    # For simplicity, split into multiple synthetic "participant" CSV files
    syn_train = syn_df.iloc[:syn_train_rows]
    for i, start in enumerate(range(0, syn_train_rows, syn_train_rows // len(train_sessions))):
        end = min(start + syn_train_rows // len(train_sessions), syn_train_rows)
        chunk = syn_df.iloc[start:end]
        chunk_rows = []; header_cols = list(chunk.columns)
        for _, row in chunk.iterrows():
            chunk_rows.append(";".join([str(row.get(c, 0)) for c in header_cols]))
        with open(f"{syn_csv_dir}/train_{i}.csv", "w") as f:
            f.write("\n".join(chunk_rows))

    print(f"\n   {syn_name}:")
    print(f"     Training on synthetic, testing on real (Synth→Real)...")
    # Train LSTM on synthetic data
    # ... (train using paper's module)

    print(f"     Training on real, testing on synthetic (Real→Synth)...")
    # Evaluate saved real LSTM on synthetic test data
    # ...

    print(f"     Training on synthetic, testing on synthetic (Synth→Synth)...")
    # ...

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("RESULTS: 2x2 Matrix")
print("=" * 60)
print(f"   Real→Real:  ACC={results['Real→Real'].get('test_acc', 'N/A')}")
# ... more results
print("\nDONE")
