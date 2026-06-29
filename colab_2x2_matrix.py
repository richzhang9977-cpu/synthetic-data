"""Colab-ready 2x2 matrix: indoor_frames + paper LSTM + SDV synthesis
Runs entirely on Colab — no local CSV export needed.
"""
# %% [markdown]
# # 2×2 Matrix: SDV Synthetic Data Evaluation with Paper's LSTM
# ## AutoTherm indoor_frames · Thermal Comfort Classification
#
# | | Test on Real | Test on Synthetic |
# |---|---|---|
# | **Train on Real** | Real→Real (baseline) | Real→Synth (TRTS) |
# | **Train on Synth** | Synth→Real (TSTR) | Synth→Synth |

# %% [markdown]
# ## 1. Install & Imports

# %%
# @title Install dependencies (~2 min)
import sys, subprocess
for pkg in ["sdv>=1.10.0", "datasets", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn", "tqdm"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("Done!")

# %%
import json, time, os, re, warnings
import numpy as np; import pandas as pd
import torch; import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datasets import load_dataset
import matplotlib.pyplot as plt; import seaborn as sns
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch {torch.__version__} | Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# %% [markdown]
# ## 2. Paper's LSTM Architecture (Section 7.7)

# %%
class ThermalComfortLSTM(nn.Module):
    """Exact architecture from az16/thermal-comfort-classification:
    LSTM(hidden=128, layers=2, dropout=0.5) → FC(64) → FC(n_classes) → Sigmoid
    """
    def __init__(self, num_features, num_classes, hidden_dim=128, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.lstm = nn.LSTM(num_features, hidden_dim, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.lstm.flatten_parameters()
        feat, _ = self.lstm(x)
        feat = feat[:, -1]
        if self.n_layers == 1: feat = self.dropout(feat)
        feat = self.fc1(feat); feat = self.dropout(feat)
        feat = self.fc2(feat)
        return self.sigmoid(feat)

# %% [markdown]
# ## 3. Load indoor_frames & Build Sequences
# Uses streaming — fast on Colab's network

# %%
# Config
PAPER_FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature",
                   "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"
SEQUENCE_WINDOW = 10
SCALE = 7
MAX_FRAMES = 80000  # enough for multiple sessions on Colab

print(f"Loading indoor_frames (up to {MAX_FRAMES} frames)...")
ds = load_dataset("kopetri/AutoTherm", "indoor_frames", streaming=True, split="train")

# Parse frames, group by recording session
sessions = {}
frame_count = 0
for sample in tqdm(ds, total=MAX_FRAMES, desc="Parsing frames"):
    try:
        meta = json.loads(sample["jpg.json"].decode("utf-8")
                        if isinstance(sample["jpg.json"], bytes) else sample["jpg.json"])
    except: continue

    fn = meta.get("file_name", "")
    session = fn.split("/")[0] if "/" in fn else "unknown"

    row = {f: meta.get(f) for f in PAPER_FEATURES}
    row[TARGET] = int(meta.get(TARGET, 0))
    row["_frame"] = frame_count  # preserve temporal order

    sessions.setdefault(session, []).append(row)
    frame_count += 1
    if frame_count >= MAX_FRAMES: break

# Filter sessions with enough frames
valid = {k: v for k, v in sessions.items() if len(v) >= SEQUENCE_WINDOW * 3}
for sess in valid:
    valid[sess].sort(key=lambda f: f["_frame"])

all_rows = [f for v in valid.values() for f in v]
df = pd.DataFrame(all_rows)
print(f"{len(valid)} sessions, {len(df)} total frames")
print(f"Label distribution: {dict(sorted(df[TARGET].value_counts().to_dict().items()))}")

# %% [markdown]
# ## 4. Per-Participant Split & Sequence Building

# %%
# Preprocess
le = LabelEncoder(); le.fit(sorted(df[TARGET].unique()))
df["_y"] = le.transform(df[TARGET])
scaler = StandardScaler()
df[PAPER_FEATURES] = scaler.fit_transform(df[PAPER_FEATURES].fillna(0))

n_classes = len(le.classes_)
n_features = len(PAPER_FEATURES)
print(f"Features: {n_features} | Classes: {n_classes} | Labels: {list(le.classes_)}")

# Split by session (per-participant)
sess_ids = sorted(valid.keys())
np.random.seed(42); np.random.shuffle(sess_ids)
n_tr = max(1, int(len(sess_ids) * 0.7))
n_vl = max(1, int(len(sess_ids) * 0.15))
train_sess = set(sess_ids[:n_tr])
val_sess = set(sess_ids[n_tr:n_tr+n_vl])
test_sess = set(sess_ids[n_tr+n_vl:])
print(f"Split: {len(train_sess)} train / {len(val_sess)} val / {len(test_sess)} test sessions")

# Build sequences
def build_sequences(df, session_set, seq_len=SEQUENCE_WINDOW):
    X_list, y_list = [], []
    for sess in session_set:
        frames = df[df["_session"] == sess] if "_session" in df.columns else \
                  [r for s, fs in valid.items() for r in fs if s == sess]
        if isinstance(frames, pd.DataFrame):
            frames = frames.reset_index(drop=True)
        else:
            frames = pd.DataFrame(frames)

        if len(frames) < seq_len: continue
        vals = frames[PAPER_FEATURES].values.astype(np.float32)
        labels = frames["_y"].values.astype(np.int64)
        for i in range(0, len(vals) - seq_len + 1, seq_len // 2):  # stride = seq_len/2
            X_list.append(vals[i:i+seq_len])
            y_list.append(labels[min(i+seq_len-1, len(labels)-1)])
    if not X_list: return np.array([]).reshape(0,1,1), np.array([])
    return np.array(X_list), np.array(y_list)

# Rebuild df with _session column from the session mapping
frame_to_sess = {}
for sess, frames in valid.items():
    for f in frames:
        frame_to_sess[f["_frame"]] = sess
df["_session"] = df["_frame"].map(frame_to_sess)

X_train, y_train = build_sequences(df, train_sess)
X_val, y_val = build_sequences(df, val_sess)
X_test, y_test = build_sequences(df, test_sess)

print(f"Sequences: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
if len(X_test) > 0:
    print(f"Test labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# %% [markdown]
# ## 5. Train LSTM (Paper Hyperparams)

# %%
def train_lstm(model, X_tr, y_tr, X_val, y_val, epochs=2, batch_size=4):
    """Paper's training: Adam(1e-4), MSE loss, LambdaLR scheduler"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    crit = nn.MSELoss()
    ds_tr = TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr))
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    amp = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for ep in range(epochs):
        model.train(); total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE); opt.zero_grad()
            if amp:
                with torch.amp.autocast("cuda"):
                    pred = model(bx)
                    loss = crit(pred, nn.functional.one_hot(by.long(), n_classes).float())
                amp.scale(loss).backward(); amp.step(opt); amp.update()
            else:
                pred = model(bx)
                loss = crit(pred, nn.functional.one_hot(by.long(), n_classes).float())
                loss.backward(); opt.step()
            sched.step(); total_loss += loss.item()
        # Eval
        model.eval()
        with torch.no_grad():
            vp = model(torch.FloatTensor(X_val).to(DEVICE)).argmax(-1).cpu()
            acc = (vp.numpy() == y_val).mean()
            f1 = f1_score(y_val, vp.numpy(), average="macro", zero_division=0)
        print(f"  Epoch {ep+1}: loss={total_loss/len(loader):.4f} | val_acc={acc:.4f} | val_f1={f1:.4f}")

def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_te).to(DEVICE)).argmax(-1).cpu().numpy()
    return {
        "acc": accuracy_score(y_te, pred),
        "f1": f1_score(y_te, pred, average="macro", zero_division=0)
    }

# %% [markdown]
# ## 6. Real→Real Baseline

# %%
print("="*50)
print("REAL -> REAL (Baseline)")
print("="*50)

lstm_real = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(lstm_real, X_train, y_train, X_val, y_val, epochs=2)
real_real = evaluate(lstm_real, X_test, y_test)
print(f"\nReal->Real: Acc={real_real['acc']:.4f} | F1={real_real['f1']:.4f}")
print(classification_report(y_test, lstm_real(torch.FloatTensor(X_test).to(DEVICE)).argmax(-1).cpu().numpy(),
                            target_names=[str(c) for c in le.classes_], zero_division=0))

# %% [markdown]
# ## 7. Generate SDV Synthetic Data (from training sessions only)

# %%
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Collect training frames
train_rows = [f for s in train_sess for f in valid[s]]
train_flat = pd.DataFrame(train_rows)[PAPER_FEATURES + [TARGET]]
train_flat[TARGET] = train_flat[TARGET].astype(str)

meta = SingleTableMetadata(); meta.detect_from_dataframe(train_flat)
meta.update_column(column_name=TARGET, sdtype="categorical")
for c in PAPER_FEATURES: meta.update_column(column_name=c, sdtype="numerical")
try: meta.remove_primary_key()
except: pass

synth_data = {}
for name, cls, ep in [("GaussianCopula", GaussianCopulaSynthesizer, 0),
                       ("CTGAN", CTGANSynthesizer, 300),
                       ("TVAE", TVAESynthesizer, 300)]:
    print(f"\n{name}: generating {len(train_flat)} rows..."); t0 = time.time()
    s = cls(meta, epochs=ep, cuda=True, verbose=False) if ep > 0 else cls(meta)
    s.fit(train_flat); df_s = s.sample(num_rows=len(train_flat))
    synth_data[name] = df_s
    print(f"  Done in {time.time()-t0:.1f}s ({df_s.shape})")

# %% [markdown]
# ## 8. Build Synthetic Sequences for 2×2 Matrix
# Split synthetic data into train/test portions (independent from real split)

# %%
def build_synth_sequences(syn_df, scaler, seq_len, n_train=None):
    """Convert flat synthetic DataFrame to train/test sequence sets"""
    X = scaler.transform(syn_df[PAPER_FEATURES].astype(float))
    y_raw = syn_df[TARGET].astype(int)
    valid_mask = (y_raw >= 0) & (y_raw < n_classes)
    X, y_raw = X[valid_mask], y_raw[valid_mask]
    y = np.clip(y_raw.values, 0, n_classes-1)

    n_total = (len(y) // seq_len) - 1
    if n_total < 20: return None, None, None, None

    X_seq = X[:n_total*seq_len].reshape(n_total, seq_len, -1)
    y_seq = y[seq_len-1::seq_len][:n_total]

    if n_train is None: n_train = int(n_total * 0.7)
    return (X_seq[:n_train], y_seq[:n_train],
            X_seq[n_train:], y_seq[n_train:])

synth_seqs = {}
for name, syn_df in synth_data.items():
    xtr, ytr, xte, yte = build_synth_sequences(syn_df, scaler, SEQUENCE_WINDOW)
    if xtr is not None:
        synth_seqs[name] = (xtr, ytr, xte, yte)
        print(f"{name}: {len(xtr)} train seqs, {len(xte)} test seqs")
    else:
        print(f"{name}: SKIP (too few sequences)")
        synth_seqs[name] = None

# %% [markdown]
# ## 9. 2×2 Matrix — All Four Cells

# %%
results = {"Real→Real": real_real}

for syn_name, seqs in synth_seqs.items():
    if seqs is None: continue
    X_syn_tr, y_syn_tr, X_syn_te, y_syn_te = seqs

    print(f"\n{'='*50}")
    print(f"{syn_name}")
    print(f"{'='*50}")

    # --- Synth→Real ---
    print(f"  Synth→Real: train LSTM on synthetic, test on real...")
    m1 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m1, X_syn_tr, y_syn_tr, X_val, y_val, epochs=1)
    results[f"{syn_name}→Real"] = evaluate(m1, X_test, y_test)
    print(f"  Synth→Real: Acc={results[f'{syn_name}→Real']['acc']:.4f} | F1={results[f'{syn_name}→Real']['f1']:.4f}")

    # --- Real→Synth ---
    print(f"  Real→Synth: use real-trained LSTM, test on synthetic...")
    results[f"Real→{syn_name}"] = evaluate(lstm_real, X_syn_te, y_syn_te)
    print(f"  Real→Synth: Acc={results[f'Real→{syn_name}']['acc']:.4f} | F1={results[f'Real→{syn_name}']['f1']:.4f}")

    # --- Synth→Synth ---
    print(f"  Synth→Synth: train LSTM on synthetic train, test on synthetic test...")
    m2 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m2, X_syn_tr, y_syn_tr, X_syn_te, y_syn_te, epochs=1)
    results[f"{syn_name}→{syn_name}"] = evaluate(m2, X_syn_te, y_syn_te)
    print(f"  Synth→Synth: Acc={results[f'{syn_name}→{syn_name}']['acc']:.4f} | F1={results[f'{syn_name}→{syn_name}']['f1']:.4f}")

# %% [markdown]
# ## 10. Final Results

# %%
print("\n" + "=" * 70)
print("2×2 MATRIX — FINAL RESULTS")
print("=" * 70)

method_names = ["Real"] + list(synth_seqs.keys())
print(f"\n{'':>25s} | {'Test on Real':>16s} | {'Test on Synthetic':>20s}")
print(f"{'':>25s} | {'F1':>7s} {'Acc':>7s} | {'F1':>7s} {'Acc':>7s}")
print("-" * 70)

# Real row
rr = results.get("Real→Real", {"f1":0,"acc":0})
first_syn = list(synth_seqs.keys())[0] if synth_seqs else None
rs = results.get(f"Real→{first_syn}", {"f1":0,"acc":0}) if first_syn else {"f1":0,"acc":0}
print(f"{'Train on Real':>25s} | {rr['f1']:7.4f} {rr['acc']:7.4f} | {rs['f1']:7.4f} {rs['acc']:7.4f}")

# Synthesis rows
for syn_name in synth_seqs:
    sr = results.get(f"{syn_name}→Real", {"f1":0,"acc":0})
    ss = results.get(f"{syn_name}→{syn_name}", {"f1":0,"acc":0})
    print(f"{'Train on '+syn_name:>25s} | {sr['f1']:7.4f} {sr['acc']:7.4f} | {ss['f1']:7.4f} {ss['acc']:7.4f}")

# Bar chart
fig, ax = plt.subplots(figsize=(10, 5))
cells = []
for row_name in ["Real"] + list(synth_seqs.keys()):
    for col_name, col_label in [("Real", "Real"), ("Synth", "Synth")]:
        key = f"{row_name}→{col_name}" if row_name != "Real" or col_name != "Real" else "Real→Real"
        if row_name == "Real" and col_name == "Synth":
            key = f"Real→{first_syn}" if first_syn else None
            label = f"Real→Synth\n({first_syn})"
        elif row_name != "Real" and col_name == "Real":
            key = f"{row_name}→Real"; label = f"{row_name}→Real"
        elif row_name == "Real" and col_name == "Real":
            key = "Real→Real"; label = "Real→Real"
        else:
            key = f"{row_name}→{row_name}"; label = f"{row_name}→Synth"

        if key and key in results:
            cells.append({"Label": label, "F1": results[key]["f1"], "Acc": results[key]["acc"]})

df_plot = pd.DataFrame(cells)
x = np.arange(len(df_plot))
w = 0.35
ax.bar(x-w/2, df_plot["F1"], w, label="F1", color="#4dabf7")
ax.bar(x+w/2, df_plot["Acc"], w, label="Accuracy", color="#ff6b6b")
ax.set_xticks(x); ax.set_xticklabels(df_plot["Label"], fontsize=9, rotation=15)
ax.set_ylabel("Score"); ax.set_title("2×2 Matrix: SDV Synthetic vs Real Data (LSTM)", fontweight="bold")
ax.legend(); ax.set_ylim(0, 1.0)
ax.axhline(y=0.591, color="green", linestyle="--", alpha=0.5, label="Paper LSTM baseline (59.1%)")
plt.tight_layout(); plt.show()

print("\nDONE")
