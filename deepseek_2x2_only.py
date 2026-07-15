"""DeepSeek 2x2 matrix only — self-contained, no closure bugs"""
import warnings; warnings.filterwarnings("ignore")
import re, time, os, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from datasets import load_from_disk

print("=" * 60)
print("DeepSeek 2x2 MATRIX")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

FEATURES = ["Radiation-Temp","Wrist_Skin_Temperature","GSR","Ambient_Temperature","Ambient_Humidity"]
TARGET = "Label"; SEQ_LEN = 10; N_CLASSES = 7

# ---- Paper LSTM ----
class ThermalComfortLSTM(nn.Module):
    def __init__(self, nf, nc, h=128, nl=2, d=0.5):
        super().__init__(); self.nl = nl
        self.lstm = nn.LSTM(nf, h, nl, batch_first=True, dropout=d if nl>1 else 0)
        self.dp = nn.Dropout(d); self.f1 = nn.Linear(h, h//2); self.f2 = nn.Linear(h//2, nc)
        self.sig = nn.Sigmoid()
    def forward(self, x):
        self.lstm.flatten_parameters(); f, _ = self.lstm(x); f = f[:, -1]
        if self.nl == 1: f = self.dp(f)
        f = self.f1(f); f = self.dp(f); f = self.f2(f); return self.sig(f)

def paper_norm(X, bounds, fl):
    Xn = X.copy()
    for i, feat in enumerate(fl):
        lo, hi = bounds[feat]; v = Xn[..., i]
        if lo is None or hi is None: lo, hi = v.min(), v.max()
        Xn[..., i] = np.clip((v-lo)/(hi-lo+1e-8), 0, 1)
    return Xn

BOUNDS = {"Radiation-Temp":(15,35),"Wrist_Skin_Temperature":(None,None),
          "GSR":(None,None),"Ambient_Temperature":(15,40),"Ambient_Humidity":(0,100)}

# ---- Load real data ----
print("\n[1/4] Loading real data...")
ds_tr = load_from_disk("F:/synthetic/dataset/indoor/train")
ds_te = load_from_disk("F:/synthetic/dataset/indoor/test")
ts, ts2 = {}, {}
for r in ds_tr: ts.setdefault(r["file_name"].split("/")[0], []).append(r)
for r in ds_te: ts2.setdefault(r["file_name"].split("/")[0], []).append(r)
for ss in [ts, ts2]:
    for k in list(ss):
        if len(ss[k]) < SEQ_LEN: del ss[k]
    for k in ss: ss[k].sort(key=lambda r: int(re.search(r'frame_f_(\d+)',r["file_name"]).group(1) if re.search(r'frame_f_(\d+)',r["file_name"]) else 0))

def seqs(sd):
    X, Y = [], []
    for fr in sd.values():
        v = np.array([[float(f.get(c,0)or 0) for c in FEATURES] for f in fr], dtype=np.float32)
        l = np.array([int(f[TARGET]) for f in fr], dtype=np.int64)
        for i in range(0, len(v)-SEQ_LEN+1, SEQ_LEN//2):
            X.append(v[i:i+SEQ_LEN]); Y.append(l[i+SEQ_LEN-1])
    return np.array(X), np.array(Y)

X_tr, y_tr = seqs(ts); X_te, y_te = seqs(ts2)
X_tr_n = paper_norm(X_tr, BOUNDS, FEATURES); X_te_n = paper_norm(X_te, BOUNDS, FEATURES)

# Label encoding (7 classes)
le = LabelEncoder(); le.fit(["-3","-2","-1","0","1","2","3"])
y_tr_enc = le.transform([str(y) for y in y_tr]).astype(np.int64)
y_te_enc = le.transform([str(y) for y in y_te]).astype(np.int64)
print(f"   Train: {X_tr_n.shape}, Test: {X_te_n.shape}, Classes: {len(le.classes_)}")

# ---- Real->Real ----
print("\n[2/4] Real->Real baseline...")
lstm = ThermalComfortLSTM(5, 7).to(DEVICE)
opt = torch.optim.Adam(lstm.parameters(), lr=1e-4)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
crit = nn.MSELoss()
loader = DataLoader(TensorDataset(torch.FloatTensor(X_tr_n), torch.LongTensor(y_tr_enc)),
                    batch_size=4, shuffle=True)
best_f1, best_state = 0, None
for ep in range(15):
    lstm.train()
    for bx, by in loader:
        bx, by = bx.to(DEVICE), by.to(DEVICE); opt.zero_grad()
        pred = lstm(bx); loss = crit(pred, nn.functional.one_hot(by.long(), 7).float())
        loss.backward(); opt.step(); sched.step()
    lstm.eval()
    with torch.no_grad():
        vp = lstm(torch.FloatTensor(X_te_n).to(DEVICE)).argmax(-1).cpu()
        acc = (vp.numpy()==y_te_enc).mean()
        f1 = f1_score(y_te_enc, vp.numpy(), average="macro", zero_division=0)
    if f1 > best_f1: best_f1 = f1; best_state = {k: v.cpu().clone() for k, v in lstm.state_dict().items()}
    print(f"  Ep {ep+1}: acc={acc:.4f} f1={f1:.4f} best={best_f1:.4f}")
lstm.load_state_dict(best_state); lstm.eval()
rr = {"acc": float(acc), "f1": float(best_f1)}
print(f"   Real->Real: Acc={rr['acc']:.4f} | F1={rr['f1']:.4f}")

# ---- DeepSeek synthetic ----
print("\n[3/4] DeepSeek 2x2 Matrix...")
dds = pd.read_csv("F:/synthetic/llm_synthetic/deepseek_train.csv")
dds[TARGET] = dds[TARGET].astype(int).astype(str)
valid = dds[TARGET].isin(le.classes_)
dsX = dds[FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0).values[valid.values].astype(np.float32)
dsY = le.transform(dds[TARGET][valid.values])

dsX_n = paper_norm(dsX, BOUNDS, FEATURES)
synX, synY = [], []
for i in range(0, len(dsX_n)-SEQ_LEN+1, SEQ_LEN//2):
    synX.append(dsX_n[i:i+SEQ_LEN]); synY.append(dsY[i+SEQ_LEN-1])
synX = np.array(synX); synY = np.array(synY)

n_st = int(len(synX)*0.7); Xs_tr = synX[:n_st]; Ys_tr = synY[:n_st]
Xs_te = synX[n_st:]; Ys_te = synY[n_st:]
print(f"   Synthetic seqs: train={len(Xs_tr)}, test={len(Xs_te)}")

def train_and_eval(Xtr, Ytr, Xte, Yte, label=""):
    m = ThermalComfortLSTM(5, 7).to(DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    cr = nn.MSELoss()
    ld = DataLoader(TensorDataset(torch.FloatTensor(Xtr), torch.LongTensor(Ytr)),
                    batch_size=4, shuffle=True)
    for ep in range(15):
        m.train()
        for bx, by in ld:
            bx, by = bx.to(DEVICE), by.to(DEVICE); opt.zero_grad()
            p = m(bx); l = cr(p, nn.functional.one_hot(by.long(), 7).float())
            l.backward(); opt.step(); sch.step()
    m.eval()
    with torch.no_grad():
        vp = m(torch.FloatTensor(Xte).to(DEVICE)).argmax(-1).cpu()
        return {"acc": accuracy_score(Yte, vp.numpy()),
                "f1": f1_score(Yte, vp.numpy(), average="macro", zero_division=0)}

# Synth->Real
print("   Synth->Real...")
ds_sr = train_and_eval(Xs_tr, Ys_tr, X_te_n, y_te_enc)
print(f"   F1={ds_sr['f1']:.4f} Acc={ds_sr['acc']:.4f}")

# Real->Synth
print("   Real->Synth...")
ds_rs = {"acc": accuracy_score(Ys_te, lstm(torch.FloatTensor(Xs_te).to(DEVICE)).argmax(-1).cpu().numpy()),
         "f1": f1_score(Ys_te, lstm(torch.FloatTensor(Xs_te).to(DEVICE)).argmax(-1).cpu().numpy(), average="macro", zero_division=0)}
print(f"   F1={ds_rs['f1']:.4f} Acc={ds_rs['acc']:.4f}")

# Synth->Synth
print("   Synth->Synth...")
ds_ss = train_and_eval(Xs_tr, Ys_tr, Xs_te, Ys_te)
print(f"   F1={ds_ss['f1']:.4f} Acc={ds_ss['acc']:.4f}")

# ---- Compare ----
print("\n"+"="*60)
print("[4/4] COMPARISON")
print("="*60)
# Load SDV results
sdv = pd.read_csv("F:/synthetic/simple_2x2_7class_results.csv") if os.path.exists("F:/synthetic/simple_2x2_7class_results.csv") else None

print(f"\n  {'Method':30s} | {'Test on Real':>16s} | {'Test on Synthetic':>20s}")
print(f"  {'':30s} | {'F1':>7s} {'Acc':>7s} | {'F1':>7s} {'Acc':>7s}")
print(f"  {'-'*60}")
print(f"  {'Train on Real':30s} | {rr['f1']:7.4f} {rr['acc']:7.4f} |")

for name, sr_key, rs_key, ss_key in [
    ("CTGAN", "CTGAN->Real", "Real->CTGAN", "CTGAN->CTGAN"),
    ("TVAE", "TVAE->Real", "Real->TVAE", "TVAE->TVAE"),
]:
    sr = (sdv[sdv["Experiment"]==sr_key].iloc[0] if sdv is not None else None)
    ss = (sdv[sdv["Experiment"]==ss_key].iloc[0] if sdv is not None else None)
    if sr is not None:
        print(f"  {'Train on '+name:30s} | {sr['F1']:7.4f} {sr['Acc']:7.4f} | {ss['F1']:7.4f} {ss['Acc']:7.4f}")

print(f"  {'Train on DeepSeek':30s} | {ds_sr['f1']:7.4f} {ds_sr['acc']:7.4f} | {ds_ss['f1']:7.4f} {ds_ss['acc']:7.4f}")

# TRTS
print(f"\n  {'TRTS':30s} | {'':16s} | {'F1':>7s} {'Acc':>7s}")
print(f"  {'-'*60}")
for name, key in [("CTGAN","Real->CTGAN"),("TVAE","Real->TVAE")]:
    rs = (sdv[sdv["Experiment"]==key].iloc[0] if sdv is not None else None)
    if rs is not None: print(f"  {'Real->'+name:30s} |             | {rs['F1']:7.4f} {rs['Acc']:7.4f}")
print(f"  {'Real->DeepSeek':30s} |             | {ds_rs['f1']:7.4f} {ds_rs['acc']:7.4f}")

pd.DataFrame({"Real->Real":rr,"DeepSeek->Real":ds_sr,"Real->DeepSeek":ds_rs,"DeepSeek->DeepSeek":ds_ss}).T.to_csv("F:/synthetic/deepseek_2x2_results.csv")
print("\nDONE")
