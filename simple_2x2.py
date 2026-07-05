# %% [markdown]
# # 2x2 Matrix: indoor + sequential LSTM + SDV
# ## Install dependencies (Colab only)

# %%
# @title pip install (skip if already installed)
import subprocess, sys
for pkg in ["sdv", "datasets", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"]:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("Done!")

# %%
"""One-shot 2x2 matrix: indoor subset + paper LSTM + SDV"""
import warnings; warnings.filterwarnings("ignore")
import time, os, re, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datasets import load_from_disk

print("=" * 60)
print("2x2 MATRIX -- indoor (local) + Paper LSTM + SDV")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# 1. Paper's LSTM
# ============================================================
class ThermalComfortLSTM(nn.Module):
    def __init__(self, n_feat, n_cls, hidden=128, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, hidden // 2)
        self.fc2 = nn.Linear(hidden // 2, n_cls)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.lstm.flatten_parameters()
        feat, _ = self.lstm(x)
        feat = feat[:, -1]
        if self.n_layers == 1: feat = self.dropout(feat)
        feat = self.fc1(feat); feat = self.dropout(feat)
        feat = self.fc2(feat)
        return self.sigmoid(feat)

def train_lstm(model, X_tr, y_tr, X_val, y_val, epochs=15, quiet=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                        batch_size=4, shuffle=True)  # paper's batch size
    best_f1, best_state = 0, None
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE); opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, nn.functional.one_hot(by.long(), n_classes).float())
            loss.backward(); opt.step(); sched.step()
        if not quiet and X_val is not None and len(X_val) > 0:
            model.eval()
            with torch.no_grad():
                vp = model(torch.FloatTensor(X_val).to(DEVICE)).argmax(-1).cpu()
                acc = (vp.numpy() == y_val).mean()
                f1 = f1_score(y_val, vp.numpy(), average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Ep {ep+1}: val_acc={acc:.4f} val_f1={f1:.4f}  best={best_f1:.4f}")
    if best_state: model.load_state_dict(best_state)  # restore best epoch (paper convention)
    model.eval(); return model

def evaluate(model, X_te, y_te):
    if len(X_te) == 0: return {"acc": 0.0, "f1": 0.0}
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_te).to(DEVICE)).argmax(-1).cpu().numpy()
    return {"acc": accuracy_score(y_te, pred),
            "f1": f1_score(y_te, pred, average="macro", zero_division=0)}

# ============================================================
# 2. Load indoor dataset (tabular frames, no images, fast)
# ============================================================
print("\n[1/6] Loading indoor data & building sequences...")
FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature", "GSR",
            "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"
SEQ_LEN = 10
N_SDV = 50000  # SDV trains on a sample (CTGAN O(N^2) complexity)

# Load both splits: train for LSTM training, test for evaluation
ds_train = load_from_disk("dataset/indoor/train")
ds_test  = load_from_disk("dataset/indoor/test")

# Group train frames by session
train_sessions = {}
for r in ds_train:
    fn = r.get("file_name", "")
    session = fn.split("/")[0] if "/" in fn else "unknown"
    train_sessions.setdefault(session, []).append(r)

# Group test frames by session
test_sessions = {}
for r in ds_test:
    fn = r.get("file_name", "")
    session = fn.split("/")[0] if "/" in fn else "unknown"
    test_sessions.setdefault(session, []).append(r)

# Combine for session listing
sessions = {**train_sessions, **test_sessions}

# Filter & sort each split independently
for ss in [train_sessions, test_sessions]:
    for k in list(ss.keys()):
        if len(ss[k]) < SEQ_LEN: del ss[k]
    for k in ss:
        ss[k].sort(key=lambda r: int(re.search(r'frame_f_(\d+)', r["file_name"]).group(1)
                                      if re.search(r'frame_f_(\d+)', r["file_name"]) else 0))

total = sum(len(v) for v in train_sessions.values()) + sum(len(v) for v in test_sessions.values())
print(f"   {len(train_sessions)} train + {len(test_sessions)} test sessions, {total} frames")

# ============================================================
# 3. Split + build sequences
# ============================================================
print("\n[2/6] Splitting & building sequences...")

# Use official paper train/test splits
train_sess = set(train_sessions.keys())
test_sess = set(test_sessions.keys())
split_mode = "official-split"
print(f"   Mode: {split_mode} | Train sessions: {len(train_sess)} | Test sessions: {len(test_sess)}")

def build_sequences(sessions_dict, session_set):
    Xl, yl = [], []
    for sess in session_set:
        frames = sessions_dict[sess]
        if len(frames) < SEQ_LEN: continue
        vals = np.array([[float(f.get(c, 0) or 0) for c in FEATURES] for f in frames], dtype=np.float32)
        labs = np.array([int(f[TARGET]) for f in frames], dtype=np.int64)
        for i in range(0, len(vals) - SEQ_LEN + 1, SEQ_LEN // 2):
            Xl.append(vals[i:i+SEQ_LEN])
            yl.append(labs[i+SEQ_LEN-1])
    if not Xl:
        return np.array([]).reshape(0, SEQ_LEN, len(FEATURES)), np.array([])
    return np.array(Xl), np.array(yl)

X_train, y_train = build_sequences(train_sessions, train_sess)
X_test, y_test = build_sequences(test_sessions, test_sess)

# For SDV: flatten ALL training frames into a static table
train_frames_list = []
for sess in train_sess:
    train_frames_list.extend(train_sessions[sess])

train_flat = pd.DataFrame(train_frames_list)[FEATURES + [TARGET]]
train_flat = train_flat.apply(pd.to_numeric, errors="coerce").fillna(0)
train_flat[TARGET] = train_flat[TARGET].astype(int)

# Fit LabelEncoder on ALL labels (train + test)
all_labels = set(int(f[TARGET]) for f in train_frames_list)
for sess in test_sess:
    for f in test_sessions[sess]:
        all_labels.add(int(f[TARGET]))
all_labels = sorted(all_labels)

le = LabelEncoder(); le.fit([str(l) for l in all_labels])
n_classes = len(le.classes_); n_features = len(FEATURES)

# Encode sequence labels (simple, no unseen labels possible now)
y_train_enc = le.transform([str(y) for y in y_train]).astype(np.int64) if len(y_train) > 0 else np.array([], dtype=np.int64)
y_test_enc = le.transform([str(y) for y in y_test]).astype(np.int64) if len(y_test) > 0 else np.array([], dtype=np.int64)

# Paper's preprocessing: min-max normalization to [0,1] with domain boundaries
PAPER_BOUNDS = {
    "Radiation-Temp": (15.0, 35.0),
    "Wrist_Skin_Temperature": (None, None),  # auto from data
    "GSR": (None, None),
    "Ambient_Temperature": (15.0, 40.0),
    "Ambient_Humidity": (0.0, 100.0),
}

def paper_normalize(X, bounds_dict, feature_list):
    """Min-max normalization to [0,1], using paper's domain boundaries where specified."""
    X_norm = X.copy()
    for i, feat in enumerate(feature_list):
        lo, hi = bounds_dict.get(feat, (None, None))
        vals = X_norm[..., i] if X_norm.ndim == 3 else X_norm[:, i]
        if lo is None or hi is None:
            lo, hi = vals.min(), vals.max()
        X_norm[..., i] = np.clip((vals - lo) / (hi - lo + 1e-8), 0, 1)
    return X_norm

X_train_scaled = paper_normalize(X_train, PAPER_BOUNDS, FEATURES)
X_test_scaled = paper_normalize(X_test, PAPER_BOUNDS, FEATURES) if len(X_test) > 0 else np.array([]).reshape(0,SEQ_LEN,n_features)

print(f"   Sequences: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
print(f"   Features: {n_features} | Classes: {n_classes} | Labels: {list(le.classes_)}")

# ============================================================
# 4. Real->Real Baseline
# ============================================================
print("\n[3/6] Real->Real...")
lstm_real = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(lstm_real, X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, epochs=15)
real_real = evaluate(lstm_real, X_test_scaled, y_test_enc)
print(f"   Real->Real: Acc={real_real['acc']:.4f} | F1={real_real['f1']:.4f}")

# ============================================================
# 5. SDV Synthesis (on flattened training frames)
# ============================================================
print("\n[4/6] SDV synthesis (from train frames only)...")
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sdv"])
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Patch CTGAN to use single-threaded transform (avoids Windows page-file crash)
def _patched_parallel_transform(self, raw_data, column_transform_info_list):
    from ctgan.data_transformer import DataTransformer as DT
    from joblib import Parallel, delayed
    processes = []
    for info in column_transform_info_list:
        data = raw_data[[info.column_name]]
        if info.column_type == 'continuous':
            processes.append(delayed(self._transform_continuous)(info, data))
        else:
            processes.append(delayed(self._transform_discrete)(info, data))
    return Parallel(n_jobs=1)(processes)  # single-thread

from ctgan import data_transformer
data_transformer.DataTransformer._parallel_transform = _patched_parallel_transform

sdv_df = train_flat[FEATURES + [TARGET]].copy()
sdv_df = sdv_df.sample(n=min(N_SDV, len(sdv_df)), random_state=42)  # sample for SDV speed
sdv_df[TARGET] = sdv_df[TARGET].astype(str)

meta = SingleTableMetadata(); meta.detect_from_dataframe(sdv_df)
meta.update_column(column_name=TARGET, sdtype="categorical")
for c in FEATURES: meta.update_column(column_name=c, sdtype="numerical")
try: meta.remove_primary_key()
except: pass

synth_data = {}
n_syn_train = min(len(train_frames_list), N_SDV)  # limit SDV generation size
n_syn_test = min(int(n_syn_train * 0.3), 15000)    # independent test set

for name, cls, ep, kwargs in [("GaussianCopula", GaussianCopulaSynthesizer, 0, {}),
                       ("CTGAN", CTGANSynthesizer, 300, {"enforce_min_max_values": False, "cuda": True}),
                       ("TVAE", TVAESynthesizer, 300, {"enforce_min_max_values": False, "cuda": True})]:
    print(f"   {name}..."); t0 = time.time()
    s = cls(meta, epochs=ep, verbose=False, **kwargs) if ep > 0 else cls(meta)
    s.fit(sdv_df)
    # Generate TWO independent synthetic datasets
    df_s_train = s.sample(num_rows=n_syn_train)  # for training
    df_s_test = s.sample(num_rows=n_syn_test)     # for testing (independent)
    synth_data[name] = {"train": df_s_train, "test": df_s_test}
    print(f"   {name}: train={df_s_train.shape}, test={df_s_test.shape} in {time.time()-t0:.1f}s")

# ============================================================
# 6. 2x2 Matrix
# ============================================================
print("\n[5/6] 2x2 Matrix...")
results = {"Real->Real": real_real}

for syn_name, syn_dict in synth_data.items():
    print(f"\n   --- {syn_name} ---")

    syn_df_train = syn_dict["train"]
    syn_df_test = syn_dict["test"]

    # --- Build sequences from INDEPENDENT synthetic train set ---
    def syn_to_seqs(df):
        X = df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
        y_raw = df[TARGET].astype(str)
        vmask = y_raw.isin(le.classes_)
        X, y_raw = X[vmask], y_raw[vmask]
        y = le.transform(y_raw)
        X_scl = paper_normalize(X, PAPER_BOUNDS, FEATURES)
        seqs_X, seqs_y = [], []
        for i in range(0, len(X_scl) - SEQ_LEN + 1, SEQ_LEN // 2):
            seqs_X.append(X_scl[i:i+SEQ_LEN])
            seqs_y.append(y[i+SEQ_LEN-1])
        return np.array(seqs_X), np.array(seqs_y)

    X_s_tr, y_s_tr = syn_to_seqs(syn_df_train)   # independent train
    X_s_te, y_s_te = syn_to_seqs(syn_df_test)     # independent test

    if len(X_s_tr) == 0 or len(X_s_te) == 0:
        print("     SKIP: too few sequences")
        continue

    # Synth->Real
    print("     Synth->Real...")
    m1 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m1, X_s_tr, y_s_tr, None, None, epochs=15, quiet=True)
    results[f"{syn_name}->Real"] = evaluate(m1, X_test_scaled, y_test_enc)
    print(f"     F1={results[f'{syn_name}->Real']['f1']:.4f} | Acc={results[f'{syn_name}->Real']['acc']:.4f}")

    # Real->Synth
    print("     Real->Synth...")
    results[f"Real->{syn_name}"] = evaluate(lstm_real, X_s_te, y_s_te)
    print(f"     F1={results[f'Real->{syn_name}']['f1']:.4f} | Acc={results[f'Real->{syn_name}']['acc']:.4f}")

    # Synth->Synth
    print("     Synth->Synth...")
    m2 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m2, X_s_tr, y_s_tr, None, None, epochs=15, quiet=True)
    results[f"{syn_name}->{syn_name}"] = evaluate(m2, X_s_te, y_s_te)
    print(f"     F1={results[f'{syn_name}->{syn_name}']['f1']:.4f} | Acc={results[f'{syn_name}->{syn_name}']['acc']:.4f}")

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "=" * 60)
print("[6/6] 2x2 MATRIX RESULTS")
print("=" * 60)
print(f"\n{'':>30s} | {'Test on Real':>16s} | {'Test on Synthetic':>20s}")
print(f"{'':>30s} | {'F1':>7s} {'Acc':>7s} | {'F1':>7s} {'Acc':>7s}")
print("-" * 70)

syn_names = list(synth_data.keys())
rr = results["Real->Real"]
first_syn = syn_names[0]
rs0 = results.get(f"Real->{first_syn}", {"f1":0,"acc":0})
print(f"{'Train on Real':>30s} | {rr['f1']:7.4f} {rr['acc']:7.4f} | {rs0['f1']:7.4f} {rs0['acc']:7.4f}")

for sn in syn_names:
    sr = results.get(f"{sn}->Real", {"f1":0,"acc":0})
    ss = results.get(f"{sn}->{sn}", {"f1":0,"acc":0})
    print(f"{'Train on '+sn:>30s} | {sr['f1']:7.4f} {sr['acc']:7.4f} | {ss['f1']:7.4f} {ss['acc']:7.4f}")

pd.DataFrame(results).T.to_csv("simple_2x2_results.csv")
print("\nDONE")
