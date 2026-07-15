"""DeepSeek 2x2 matrix — same pipeline as simple_2x2.py, just different synthetic data"""
import warnings; warnings.filterwarnings("ignore")
import re, time, os, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk

print("=" * 60)
print("DeepSeek 2x2 MATRIX — Paper LSTM pipeline")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Exact same config as simple_2x2.py
FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature", "GSR",
            "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"; SEQ_LEN = 10; N_SDV = 50000
PAPER_BOUNDS = {
    "Radiation-Temp": (15.0, 35.0), "Wrist_Skin_Temperature": (None, None),
    "GSR": (None, None), "Ambient_Temperature": (15.0, 40.0),
    "Ambient_Humidity": (0.0, 100.0),
}

# ---- Paper LSTM ----
class ThermalComfortLSTM(nn.Module):
    def __init__(self, nf, nc, h=128, nl=2, d=0.5):
        super().__init__(); self.nl = nl
        self.lstm = nn.LSTM(nf, h, nl, batch_first=True, dropout=d if nl > 1 else 0.0)
        self.drop = nn.Dropout(d); self.fc1 = nn.Linear(h, h // 2)
        self.fc2 = nn.Linear(h // 2, nc); self.sig = nn.Sigmoid()
    def forward(self, x):
        self.lstm.flatten_parameters(); f, _ = self.lstm(x); f = f[:, -1]
        if self.nl == 1: f = self.drop(f)
        f = self.fc1(f); f = self.drop(f); f = self.fc2(f); return self.sig(f)

def train_lstm(model, Xtr, ytr, Xval, yval, epochs=15, quiet=False):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(ytr)), batch_size=4, shuffle=True)
    bf, bs = 0, None
    for ep in range(epochs):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE); opt.zero_grad()
            pred = model(bx)
            loss = crit(pred, nn.functional.one_hot(by.long(), pred.shape[-1]).float())
            loss.backward(); opt.step(); sched.step()
        if not quiet and Xval is not None and len(Xval) > 0:
            model.eval()
            with torch.no_grad():
                vp = model(torch.FloatTensor(Xval).to(DEVICE)).argmax(-1).cpu()
                acc = (vp.numpy() == yval).mean()
                f1 = f1_score(yval, vp.numpy(), average="macro", zero_division=0)
            if f1 > bf: bf = f1; bs = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"  Ep {ep+1}: val_acc={acc:.4f} val_f1={f1:.4f} best={bf:.4f}")
    if bs: model.load_state_dict(bs)
    model.eval(); return model

def evaluate(model, Xte, yte):
    if len(Xte) == 0: return {"acc": 0.0, "f1": 0.0}
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(Xte).to(DEVICE)).argmax(-1).cpu().numpy()
    return {"acc": accuracy_score(yte, pred), "f1": f1_score(yte, pred, average="macro", zero_division=0)}

def paper_normalize(X, bounds, fl):
    Xn = X.copy()
    for i, f in enumerate(fl):
        lo, hi = bounds[f]; v = Xn[..., i] if Xn.ndim == 3 else Xn[:, i]
        if lo is None or hi is None: lo, hi = v.min(), v.max()
        Xn[..., i] = np.clip((v - lo) / (hi - lo + 1e-8), 0, 1)
    return Xn

# ---- Load real data ----
print("\n[1/3] Loading real data & building sequences...")
ds_train = load_from_disk("F:/synthetic/dataset/indoor/train")
ds_test  = load_from_disk("F:/synthetic/dataset/indoor/test")

ts, ts2 = {}, {}
for r in ds_train:
    s = r["file_name"].split("/")[0]; ts.setdefault(s, []).append(r)
for r in ds_test:
    s = r["file_name"].split("/")[0]; ts2.setdefault(s, []).append(r)
for ss in [ts, ts2]:
    for k in list(ss.keys()):
        if len(ss[k]) < SEQ_LEN: del ss[k]
    for k in ss: ss[k].sort(key=lambda r: int(re.search(r'frame_f_(\d+)', r["file_name"]).group(1)
                                                  if re.search(r'frame_f_(\d+)', r["file_name"]) else 0))

def build_seqs(sd):
    X, Y = [], []
    for frames in sd.values():
        vals = np.array([[float(f.get(c, 0) or 0) for c in FEATURES] for f in frames], dtype=np.float32)
        labs = np.array([int(f[TARGET]) for f in frames], dtype=np.int64)
        for i in range(0, len(vals) - SEQ_LEN + 1, SEQ_LEN // 2):
            X.append(vals[i:i+SEQ_LEN]); Y.append(labs[i+SEQ_LEN-1])
    return np.array(X), np.array(Y)

X_train, y_train = build_seqs(ts); X_test, y_test = build_seqs(ts2)
all_labels = sorted(set(int(f[TARGET]) for v in ts.values() for f in v) |
                    set(int(f[TARGET]) for v in ts2.values() for f in v))
le = LabelEncoder(); le.fit([str(l) for l in all_labels])
n_classes = len(le.classes_); n_features = len(FEATURES)
yte = le.transform([str(y) for y in y_test]).astype(np.int64)
ytr = le.transform([str(y) for y in y_train]).astype(np.int64)
Xtr_n = paper_normalize(X_train, PAPER_BOUNDS, FEATURES)
Xte_n = paper_normalize(X_test, PAPER_BOUNDS, FEATURES)
print(f"   Train: {Xtr_n.shape}, Test: {Xte_n.shape}, Classes: {n_classes}")

# ---- Real->Real baseline ----
print("\n[2/3] Real->Real baseline...")
lstm_real = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(lstm_real, Xtr_n, ytr, Xte_n, yte, epochs=15)
real_real = evaluate(lstm_real, Xte_n, yte)
print(f"   Real->Real: Acc={real_real['acc']:.4f} | F1={real_real['f1']:.4f}")

# ---- DeepSeek 2x2 ----
print("\n[3/3] DeepSeek 2x2 Matrix...")
dds = pd.read_csv("F:/synthetic/llm_synthetic/deepseek_train.csv")
dds[TARGET] = dds[TARGET].astype(str)

# Generate independent train/test from DeepSeek data
# Use first 70% rows for train set, last 30% for test set
dsX = dds[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float32)
valid_ds = dds[TARGET].isin(le.classes_); dsX = dsX[valid_ds]; dsY_raw = dds[TARGET][valid_ds]
dsY = le.transform(dsY_raw)
dsX_n = paper_normalize(dsX, PAPER_BOUNDS, FEATURES)

# Build synthetic sequences (same sliding window as real data)
syn_seqs, syn_Y = [], []
for i in range(0, len(dsX_n) - SEQ_LEN + 1, SEQ_LEN // 2):
    syn_seqs.append(dsX_n[i:i+SEQ_LEN]); syn_Y.append(dsY[i+SEQ_LEN-1])
syn_seqs = np.array(syn_seqs); syn_Y = np.array(syn_seqs)

# Independent train/test split for synthetic data
n_st = int(len(syn_seqs) * 0.7)
Xs_tr = syn_seqs[:n_st]; Ys_tr = syn_Y[:n_st]
Xs_te = syn_seqs[n_st:]; Ys_te = syn_Y[n_st:]

print(f"   Synth seqs: train={len(Xs_tr)}, test={len(Xs_te)}")

# Synth->Real
print("   Synth->Real...")
print(f"   DEBUG: n_features={n_features}, n_classes={n_classes}, Xs_tr.shape={Xs_tr.shape}, Ys_tr[:5]={Ys_tr[:5]}, Ys_tr min/max={Ys_tr.min()}/{Ys_tr.max()}")
assert len(Xs_tr) > 0, "Empty training set"
assert Ys_tr.max() < n_classes, f"Label {Ys_tr.max()} >= n_classes {n_classes}"
m1 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(m1, Xs_tr, Ys_tr, None, None, epochs=3, quiet=True)
ds_sr = evaluate(m1, Xte_n, yte)
print(f"   Synth->Real: F1={ds_sr['f1']:.4f} | Acc={ds_sr['acc']:.4f}")

# Real->Synth
print("   Real->Synth...")
ds_rs = evaluate(lstm_real, Xs_te, Ys_te)
print(f"   Real->Synth: F1={ds_rs['f1']:.4f} | Acc={ds_rs['acc']:.4f}")

# Synth->Synth
print("   Synth->Synth...")
m2 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(m2, Xs_tr, Ys_tr, None, None, epochs=15, quiet=True)
ds_ss = evaluate(m2, Xs_te, Ys_te)
print(f"   Synth->Synth: F1={ds_ss['f1']:.4f} | Acc={ds_ss['acc']:.4f}")

# ---- Compare ----
print("\n" + "=" * 60)
print("COMPARISON: DeepSeek vs SDV (same LSTM pipeline)")
print("=" * 60)

# Load saved SDV results
sdv = {}
if os.path.exists("F:/synthetic/simple_2x2_7class_results.csv"):
    df_sdv = pd.read_csv("F:/synthetic/simple_2x2_7class_results.csv")
    for _, row in df_sdv.iterrows():
        sdv[row["Experiment"]] = (row["F1"], row["Acc"])

print(f"\n  {'Method':30s} | {'Test on Real':>16s} | {'Test on Synthetic':>20s}")
print(f"  {'':30s} | {'F1':>7s} {'Acc':>7s} | {'F1':>7s} {'Acc':>7s}")
print(f"  {'-'*60}")
rr = real_real
print(f"  {'Train on Real':30s} | {rr['f1']:7.4f} {rr['acc']:7.4f} |")

# CTGAN
ct_sr = sdv.get("CTGAN->Real", (0, 0)); ct_rs = sdv.get("Real->CTGAN", (0, 0))
ct_ss = sdv.get("CTGAN->CTGAN", (0, 0))
print(f"  {'Train on CTGAN':30s} | {ct_sr[0]:7.4f} {ct_sr[1]:7.4f} | {ct_ss[0]:7.4f} {ct_ss[1]:7.4f}")

# TVAE
tv_sr = sdv.get("TVAE->Real", (0, 0)); tv_rs = sdv.get("Real->TVAE", (0, 0))
tv_ss = sdv.get("TVAE->TVAE", (0, 0))
print(f"  {'Train on TVAE':30s} | {tv_sr[0]:7.4f} {tv_sr[1]:7.4f} | {tv_ss[0]:7.4f} {tv_ss[1]:7.4f}")

# DeepSeek
print(f"  {'Train on DeepSeek':30s} | {ds_sr['f1']:7.4f} {ds_sr['acc']:7.4f} | {ds_ss['f1']:7.4f} {ds_ss['acc']:7.4f}")

# TRTS comparison
print(f"\n  {'TRTS (Real->Synth)':30s} | {'':16s} | {'F1':>7s} {'Acc':>7s}")
print(f"  {'-'*60}")
print(f"  {'Real->CTGAN':30s} |             | {ct_rs[0]:7.4f} {ct_rs[1]:7.4f}")
print(f"  {'Real->TVAE':30s} |             | {tv_rs[0]:7.4f} {tv_rs[1]:7.4f}")
print(f"  {'Real->DeepSeek':30s} |             | {ds_rs['f1']:7.4f} {ds_rs['acc']:7.4f}")

out = pd.DataFrame({
    "Real->Real": rr, "DeepSeek->Real": ds_sr,
    "Real->DeepSeek": ds_rs, "DeepSeek->DeepSeek": ds_ss
}).T
out.to_csv("F:/synthetic/deepseek_2x2_results.csv")
print(f"\nSaved: deepseek_2x2_results.csv")
print("DONE")
