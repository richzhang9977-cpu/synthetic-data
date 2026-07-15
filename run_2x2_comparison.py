"""2x2 matrix for all synthetic methods: CTGAN, TVAE, DeepSeek"""
import warnings; warnings.filterwarnings("ignore")
import re, time, os, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk

print("=" * 60)
print("2x2 MATRIX — CTGAN vs TVAE vs DeepSeek")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# Config
# ============================================================
FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature", "GSR",
            "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"
SEQ_LEN = 10
PAPER_BOUNDS = {
    "Radiation-Temp": (15.0, 35.0),
    "Wrist_Skin_Temperature": (None, None),
    "GSR": (None, None),
    "Ambient_Temperature": (15.0, 40.0),
    "Ambient_Humidity": (0.0, 100.0),
}

def paper_normalize(X, bounds, feat_list):
    X_n = X.copy()
    for i, feat in enumerate(feat_list):
        lo, hi = bounds[feat]
        vals = X_n[..., i] if X_n.ndim == 3 else X_n[:, i]
        if lo is None or hi is None: lo, hi = vals.min(), vals.max()
        X_n[..., i] = np.clip((vals - lo) / (hi - lo + 1e-8), 0, 1)
    return X_n

# ============================================================
# Paper's LSTM
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
        feat, _ = self.lstm(x); feat = feat[:, -1]
        if self.n_layers == 1: feat = self.dropout(feat)
        feat = self.fc1(feat); feat = self.dropout(feat)
        feat = self.fc2(feat)
        return self.sigmoid(feat)

def train_lstm(model, X_tr, y_tr, X_val, y_val, epochs=15, quiet=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    crit = nn.MSELoss()
    if isinstance(X_tr, list):  # for synthetic sequences
        X_tr = np.array(X_tr)
    loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
                        batch_size=32, shuffle=True)
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
            if f1 > best_f1: best_f1 = f1; best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    model.eval(); return model

def evaluate(model, X_te, y_te):
    if len(X_te) == 0: return {"acc": 0.0, "f1": 0.0}
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_te).to(DEVICE)).argmax(-1).cpu().numpy()
    return {"acc": accuracy_score(y_te, pred),
            "f1": f1_score(y_te, pred, average="macro", zero_division=0)}

# ============================================================
# 1. Load real data & build sequences
# ============================================================
print("\n[1/4] Loading real indoor data & building sequences...")
ds_train = load_from_disk("F:/synthetic/dataset/indoor/train")
ds_test  = load_from_disk("F:/synthetic/dataset/indoor/test")

train_sessions, test_sessions = {}, {}
for r in ds_train:
    sess = r["file_name"].split("/")[0]
    train_sessions.setdefault(sess, []).append(r)
for r in ds_test:
    sess = r["file_name"].split("/")[0]
    test_sessions.setdefault(sess, []).append(r)

for ss in [train_sessions, test_sessions]:
    for k in list(ss.keys()):
        if len(ss[k]) < SEQ_LEN: del ss[k]
    for k in ss:
        ss[k].sort(key=lambda r: int(re.search(r'frame_f_(\d+)', r["file_name"]).group(1)
                                      if re.search(r'frame_f_(\d+)', r["file_name"]) else 0))

def build_sequences(sess_dict):
    Xl, yl = [], []
    for frames in sess_dict.values():
        if len(frames) < SEQ_LEN: continue
        vals = np.array([[float(f.get(c, 0) or 0) for c in FEATURES] for f in frames], dtype=np.float32)
        labs = np.array([int(f[TARGET]) for f in frames], dtype=np.int64)
        for i in range(0, len(vals) - SEQ_LEN + 1, SEQ_LEN // 2):
            Xl.append(vals[i:i+SEQ_LEN]); yl.append(labs[i+SEQ_LEN-1])
    return np.array(Xl), np.array(yl)

X_train, y_train = build_sequences(train_sessions)
X_test, y_test = build_sequences(test_sessions)

# Label encoding
all_labels = sorted(set(int(f[TARGET]) for v in train_sessions.values() for f in v) |
                    set(int(f[TARGET]) for v in test_sessions.values() for f in v))
le = LabelEncoder(); le.fit([str(l) for l in all_labels])
n_classes = len(le.classes_); n_features = len(FEATURES)

y_train_enc = le.transform([str(y) for y in y_train]).astype(np.int64)
y_test_enc = le.transform([str(y) for y in y_test]).astype(np.int64)

X_train_n = paper_normalize(X_train, PAPER_BOUNDS, FEATURES)
X_test_n = paper_normalize(X_test, PAPER_BOUNDS, FEATURES)

# Sample training sequences for speed (batch_size=32 × 313K is too slow)
np.random.seed(42)
n_sample = min(30000, len(X_train_n))
idx = np.random.choice(len(X_train_n), n_sample, replace=False)
X_train_n = X_train_n[idx]; y_train_enc = y_train_enc[idx]

print(f"   Train: {X_train_n.shape}, Test: {X_test_n.shape}")
print(f"   Classes: {n_classes} | Labels: {list(le.classes_)}")

# ============================================================
# 2. Real→Real baseline
# ============================================================
print("\n[2/4] Real->Real baseline...")
lstm_real = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(lstm_real, X_train_n, y_train_enc, X_test_n, y_test_enc, epochs=15)
real_real = evaluate(lstm_real, X_test_n, y_test_enc)
print(f"   Real->Real: Acc={real_real['acc']:.4f} | F1={real_real['f1']:.4f}")
torch.save(lstm_real.state_dict(), "F:/synthetic/lstm_real.pt")

# ============================================================
# 3. Load synthetic data + run 2x2
# ============================================================
print("\n[3/4] Loading synthetic data & running 2x2 matrix...")

results = {"Real->Real": real_real}

# Methods to evaluate
methods = {
    "CTGAN": "F:/synthetic/synth_indoor_paper_CTGAN.csv",
    "TVAE": "F:/synthetic/synth_indoor_paper_TVAE.csv",
    "DeepSeek": "F:/synthetic/llm_synthetic/deepseek_train.csv",
}

for name, path in methods.items():
    if not os.path.exists(path):
        print(f"\n   {name}: SKIP (file not found: {path})")
        continue

    print(f"\n   --- {name} ---")
    syn_df = pd.read_csv(path)
    syn_df[TARGET] = syn_df[TARGET].astype(str)

    # Preprocess
    syn_X = syn_df[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
    valid = syn_df[TARGET].isin(le.classes_)
    syn_X, syn_y_raw = syn_X[valid], syn_df[TARGET][valid]
    syn_y = le.transform(syn_y_raw)

    syn_X_n = paper_normalize(syn_X, PAPER_BOUNDS, FEATURES)

    # Build synthetic sequences
    syn_seqs, syn_y_seqs = [], []
    for i in range(0, len(syn_X_n) - SEQ_LEN + 1, SEQ_LEN // 2):
        syn_seqs.append(syn_X_n[i:i+SEQ_LEN])
        syn_y_seqs.append(syn_y[i+SEQ_LEN-1])
    syn_seqs = np.array(syn_seqs); syn_y_seqs = np.array(syn_y_seqs)

    # Split synthetic train/test (independent from real split)
    n_syn_tr = max(int(len(syn_seqs) * 0.7), SEQ_LEN)
    X_s_tr = syn_seqs[:n_syn_tr]; y_s_tr = syn_y_seqs[:n_syn_tr]
    X_s_te = syn_seqs[n_syn_tr:]; y_s_te = syn_y_seqs[n_syn_tr:]

    print(f"     Synth seqs: train={len(X_s_tr)}, test={len(X_s_te)}")

    # Synth→Real
    print("     Synth->Real...")
    m1 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m1, X_s_tr, y_s_tr, None, None, epochs=15, quiet=True)
    results[f"{name}->Real"] = evaluate(m1, X_test_n, y_test_enc)
    print(f"     F1={results[f'{name}->Real']['f1']:.4f} | Acc={results[f'{name}->Real']['acc']:.4f}")

    # Real→Synth
    print("     Real->Synth...")
    results[f"Real->{name}"] = evaluate(lstm_real, X_s_te, y_s_te)
    print(f"     F1={results[f'Real->{name}']['f1']:.4f} | Acc={results[f'Real->{name}']['acc']:.4f}")

    # Synth→Synth
    print("     Synth->Synth...")
    m2 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m2, X_s_tr, y_s_tr, None, None, epochs=15, quiet=True)
    results[f"{name}->{name}"] = evaluate(m2, X_s_te, y_s_te)
    print(f"     F1={results[f'{name}->{name}']['f1']:.4f} | Acc={results[f'{name}->{name}']['acc']:.4f}")

# ============================================================
# 4. Summary
# ============================================================
print("\n" + "=" * 60)
print("[4/4] 2x2 MATRIX COMPARISON")
print("=" * 60)

for method in ["Real"] + list(methods.keys()):
    print(f"\n{'='*50}")
    print(f"  {method}")
    print(f"{'='*50}")
    print(f"  {'':20s} | {'Test on Real':>16s} | {'Test on Synthetic':>20s}")
    print(f"  {'':20s} | {'F1':>7s} {'Acc':>7s} | {'F1':>7s} {'Acc':>7s}")
    print(f"  {'-'*50}")
    if method == "Real":
        rr = results.get("Real->Real", {"f1":0,"acc":0})
        first_other = list(methods.keys())[0]
        rs = results.get(f"Real->{first_other}", {"f1":0,"acc":0})
        print(f"  {'Train on Real':20s} | {rr['f1']:7.4f} {rr['acc']:7.4f} | {rs['f1']:7.4f} {rs['acc']:7.4f}")
    else:
        sr = results.get(f"{method}->Real", {"f1":0,"acc":0})
        rs = results.get(f"Real->{method}", {"f1":0,"acc":0})
        ss = results.get(f"{method}->{method}", {"f1":0,"acc":0})
        print(f"  {'Train on '+method:20s} | {sr['f1']:7.4f} {sr['acc']:7.4f} | {ss['f1']:7.4f} {ss['acc']:7.4f}")
        print(f"  {'Real->'+method+' (TRTS)':20s} |         | {rs['f1']:7.4f} {rs['acc']:7.4f}")

# Save
out = pd.DataFrame(results).T
out.to_csv("F:/synthetic/2x2_comparison_results.csv")
print(f"\nSaved: 2x2_comparison_results.csv")
print("DONE")
