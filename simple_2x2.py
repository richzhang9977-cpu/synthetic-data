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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datasets import load_dataset

print("=" * 60)
print("2x2 MATRIX -- indoor + Paper LSTM + SDV")
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

def train_lstm(model, X_tr, y_tr, X_val, y_val, epochs=3, quiet=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                        batch_size=32, shuffle=True)
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
            print(f"  Ep {ep+1}: val_acc={acc:.4f} val_f1={f1:.4f}")
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
N = 250000

ds = load_dataset("kopetri/AutoTherm", "indoor", streaming=True, split="train")

# Group frames by recording session (folder name in file_name)
sessions = {}
for i, r in enumerate(ds):
    if i >= N: break
    fn = r.get("file_name", "")
    session = fn.split("/")[0] if "/" in fn else f"unk_{i}"
    sessions.setdefault(session, []).append(r)

# Filter & sort
valid = {k: v for k, v in sessions.items() if len(v) >= SEQ_LEN}
for k in valid:
    valid[k].sort(key=lambda r: int(re.search(r'frame_f_(\d+)', r["file_name"]).group(1)
                                    if re.search(r'frame_f_(\d+)', r["file_name"]) else 0))

total = sum(len(v) for v in valid.values())
print(f"   {len(valid)} sessions, {total} frames")

# ============================================================
# 3. Split + build sequences
# ============================================================
print("\n[2/6] Splitting & building sequences...")

sess_ids = sorted(valid.keys())
np.random.seed(42); np.random.shuffle(sess_ids)

if len(sess_ids) > 1:
    n_tr = max(1, int(len(sess_ids) * 0.7))
    train_sess = set(sess_ids[:n_tr])
    test_sess = set(sess_ids[n_tr:])
    split_mode = "inter-session"
else:
    train_sess = test_sess = set(sess_ids)
    split_mode = "intra-session"
print(f"   Mode: {split_mode} | Train: {len(train_sess)} | Test: {len(test_sess)}")

def build_sequences(sessions_dict, session_set, is_train=True):
    Xl, yl = [], []
    for sess in session_set:
        frames = sessions_dict[sess]

        if split_mode == "intra-session":
            sp = int(len(frames) * 0.7)
            frames = frames[:sp] if is_train else frames[sp:]

        if len(frames) < SEQ_LEN: continue
        vals = np.array([[float(f.get(c, 0) or 0) for c in FEATURES] for f in frames], dtype=np.float32)
        labs = np.array([int(f[TARGET]) for f in frames], dtype=np.int64)
        for i in range(0, len(vals) - SEQ_LEN + 1, SEQ_LEN // 2):
            Xl.append(vals[i:i+SEQ_LEN])
            yl.append(labs[i+SEQ_LEN-1])
    if not Xl:
        return np.array([]).reshape(0, SEQ_LEN, len(FEATURES)), np.array([])
    return np.array(Xl), np.array(yl)

X_train, y_train = build_sequences(valid, train_sess, is_train=True)
X_test, y_test = build_sequences(valid, test_sess, is_train=False)

# For SDV: flatten ALL training frames into a static table
train_frames_list = []
for sess in train_sess:
    fr = valid[sess]
    if split_mode == "intra-session":
        fr = fr[:int(len(fr)*0.7)]
    train_frames_list.extend(fr)

train_flat = pd.DataFrame(train_frames_list)[FEATURES + [TARGET]]
train_flat = train_flat.apply(pd.to_numeric, errors="coerce").fillna(0)
train_flat[TARGET] = train_flat[TARGET].astype(int)

# Fit LabelEncoder on ALL labels (train + test)
all_labels = set(int(f[TARGET]) for f in train_frames_list)
for sess in test_sess:
    if split_mode == "intra-session":
        fr = valid[sess][int(len(valid[sess])*0.7):]
    else:
        fr = valid[sess]
    for f in fr:
        all_labels.add(int(f[TARGET]))
all_labels = sorted(all_labels)

le = LabelEncoder(); le.fit([str(l) for l in all_labels])
n_classes = len(le.classes_); n_features = len(FEATURES)

# Encode sequence labels (simple, no unseen labels possible now)
y_train_enc = le.transform([str(y) for y in y_train]).astype(np.int64) if len(y_train) > 0 else np.array([], dtype=np.int64)
y_test_enc = le.transform([str(y) for y in y_test]).astype(np.int64) if len(y_test) > 0 else np.array([], dtype=np.int64)

# Scale
scaler = StandardScaler()
train_vals = np.concatenate([s for s in X_train])
scaler.fit(train_vals)
X_train_scaled = np.array([scaler.transform(s) for s in X_train]) if len(X_train) > 0 else np.array([]).reshape(0,SEQ_LEN,n_features)
X_test_scaled = np.array([scaler.transform(s) for s in X_test]) if len(X_test) > 0 else np.array([]).reshape(0,SEQ_LEN,n_features)

print(f"   Sequences: train={X_train_scaled.shape}, test={X_test_scaled.shape}")
print(f"   Features: {n_features} | Classes: {n_classes} | Labels: {list(le.classes_)}")

# ============================================================
# 4. Real->Real Baseline
# ============================================================
print("\n[3/6] Real->Real...")
lstm_real = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(lstm_real, X_train_scaled, y_train_enc, X_test_scaled, y_test_enc, epochs=3)
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

sdv_df = train_flat[FEATURES + [TARGET]].copy()
sdv_df[TARGET] = sdv_df[TARGET].astype(str)

meta = SingleTableMetadata(); meta.detect_from_dataframe(sdv_df)
meta.update_column(column_name=TARGET, sdtype="categorical")
for c in FEATURES: meta.update_column(column_name=c, sdtype="numerical")
try: meta.remove_primary_key()
except: pass

synth_data = {}
for name, cls, ep in [("GaussianCopula", GaussianCopulaSynthesizer, 0),
                       ("CTGAN", CTGANSynthesizer, 300),
                       ("TVAE", TVAESynthesizer, 300)]:
    print(f"   {name}..."); t0 = time.time()
    s = cls(meta, epochs=ep, cuda=True, verbose=False) if ep > 0 else cls(meta)
    s.fit(sdv_df); df_s = s.sample(num_rows=len(sdv_df))
    synth_data[name] = df_s
    print(f"   {name}: {df_s.shape} in {time.time()-t0:.1f}s")

# ============================================================
# 6. 2x2 Matrix
# ============================================================
print("\n[5/6] 2x2 Matrix...")
results = {"Real->Real": real_real}

for syn_name, syn_df in synth_data.items():
    print(f"\n   --- {syn_name} ---")

    syn_X = syn_df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    syn_y_raw = syn_df[TARGET].astype(str)
    valid_mask = syn_y_raw.isin(le.classes_)
    syn_X, syn_y_raw = syn_X[valid_mask], syn_y_raw[valid_mask]
    syn_y = le.transform(syn_y_raw)

    syn_X_scaled = scaler.transform(syn_X)
    syn_seqs, syn_seqs_y = [], []
    for i in range(0, len(syn_X_scaled) - SEQ_LEN + 1, SEQ_LEN // 2):
        syn_seqs.append(syn_X_scaled[i:i+SEQ_LEN])
        syn_seqs_y.append(syn_y[i+SEQ_LEN-1])
    syn_seqs = np.array(syn_seqs); syn_seqs_y = np.array(syn_seqs_y)

    n_syn_tr = max(1, int(len(syn_seqs) * 0.7))
    X_s_tr = syn_seqs[:n_syn_tr]; y_s_tr = syn_seqs_y[:n_syn_tr]
    X_s_te = syn_seqs[n_syn_tr:]; y_s_te = syn_seqs_y[n_syn_tr:]

    # Synth->Real
    print("     Synth->Real...")
    m1 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m1, X_s_tr, y_s_tr, None, None, epochs=3, quiet=True)
    results[f"{syn_name}->Real"] = evaluate(m1, X_test_scaled, y_test_enc)
    print(f"     F1={results[f'{syn_name}->Real']['f1']:.4f} | Acc={results[f'{syn_name}->Real']['acc']:.4f}")

    # Real->Synth
    print("     Real->Synth...")
    results[f"Real->{syn_name}"] = evaluate(lstm_real, X_s_te, y_s_te)
    print(f"     F1={results[f'Real->{syn_name}']['f1']:.4f} | Acc={results[f'Real->{syn_name}']['acc']:.4f}")

    # Synth->Synth
    print("     Synth->Synth...")
    m2 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m2, X_s_tr, y_s_tr, None, None, epochs=3, quiet=True)
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
