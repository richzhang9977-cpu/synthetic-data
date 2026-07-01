"""One-shot 2x2 matrix: combined subset + paper LSTM + SDV"""
import warnings; warnings.filterwarnings("ignore")
import time, os, json, numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datasets import load_dataset

print("=" * 60)
print("2×2 MATRIX — combined + Paper LSTM + SDV")
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
        if not quiet and X_val is not None:
            model.eval()
            with torch.no_grad():
                vp = model(torch.FloatTensor(X_val).to(DEVICE)).argmax(-1).cpu()
                acc = (vp.numpy() == y_val).mean()
                f1 = f1_score(y_val, vp.numpy(), average="macro", zero_division=0)
            print(f"  Ep {ep+1}: val_acc={acc:.4f} val_f1={f1:.4f}")
    model.eval(); return model

def evaluate(model, X_te, y_te):
    model.eval()
    with torch.no_grad():
        pred = model(torch.FloatTensor(X_te).to(DEVICE)).argmax(-1).cpu().numpy()
    return {"acc": accuracy_score(y_te, pred),
            "f1": f1_score(y_te, pred, average="macro", zero_division=0)}

# ============================================================
# 2. Load Data (combined = tabular, no images, fast)
# ============================================================
print("\n[1/6] Loading data...")
FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature", "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"
N = 20000

ds = load_dataset("kopetri/AutoTherm", "combined", streaming=True, split="train")
rows = []
for i, r in enumerate(ds):
    if i >= N: break
    rows.append({f: r.get(f, 0) for f in FEATURES + [TARGET]})
df = pd.DataFrame(rows).dropna()
df[TARGET] = df[TARGET].astype(int).astype(str)

le = LabelEncoder(); le.fit(sorted(df[TARGET].unique(), key=int))
df["_y"] = le.transform(df[TARGET])
n_classes = len(le.classes_); n_features = len(FEATURES)
print(f"   {len(df)} rows | {n_features} features | {n_classes} classes | Labels: {list(le.classes_)}")
print(f"   Distribution: {dict(sorted(df[TARGET].value_counts().to_dict().items(), key=lambda x: int(x[0])))}")

# ============================================================
# 3. Split & Preprocess
# ============================================================
print("\n[2/6] Split & preprocess...")
train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[TARGET])
print(f"   Train: {len(train_df)} | Test: {len(test_df)}")

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[FEATURES].astype(float))
X_test = scaler.transform(test_df[FEATURES].astype(float))
y_train = train_df["_y"].values.astype(np.int64)
y_test = test_df["_y"].values.astype(np.int64)

# LSTM needs (B, seq_len, features); seq_len=1 for tabular data
X_train_3d = X_train.reshape(-1, 1, n_features)
X_test_3d = X_test.reshape(-1, 1, n_features)

# ============================================================
# 4. Real→Real Baseline
# ============================================================
print("\n[3/6] Real→Real...")
lstm_real = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
train_lstm(lstm_real, X_train_3d, y_train, X_test_3d, y_test, epochs=3)
real_real = evaluate(lstm_real, X_test_3d, y_test)
print(f"   Real→Real: Acc={real_real['acc']:.4f} | F1={real_real['f1']:.4f}")
torch.save(lstm_real.state_dict(), "lstm_real_model.pt")  # saves to Colab current dir

# ============================================================
# 5. SDV Synthesis
# ============================================================
print("\n[4/6] SDV synthesis (from train set only)...")
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

sdv_df = train_df[FEATURES + [TARGET]].copy()
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
# 6. 2×2 Matrix
# ============================================================
print("\n[5/6] 2×2 Matrix...")
results = {"Real→Real": real_real}

for syn_name, syn_df in synth_data.items():
    print(f"\n   --- {syn_name} ---")

    # Prepare synthetic data
    syn_clean = syn_df[FEATURES + [TARGET]].copy()
    syn_clean[TARGET] = syn_clean[TARGET].astype(str)
    valid = syn_clean[TARGET].isin(le.classes_)
    syn_clean = syn_clean[valid]

    X_syn = scaler.transform(syn_clean[FEATURES].astype(float))
    y_syn = le.transform(syn_clean[TARGET])

    # Split synthetic into train/test (independent from real split)
    n_syn_train = int(len(X_syn) * 0.7)
    X_syn_tr = X_syn[:n_syn_train].reshape(-1, 1, n_features)
    y_syn_tr = y_syn[:n_syn_train]
    X_syn_te = X_syn[n_syn_train:].reshape(-1, 1, n_features)
    y_syn_te = y_syn[n_syn_train:]

    # Synth→Real
    print(f"     Synth→Real...")
    m1 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m1, X_syn_tr, y_syn_tr, None, None, epochs=3, quiet=True)
    results[f"{syn_name}→Real"] = evaluate(m1, X_test_3d, y_test)
    print(f"     F1={results[f'{syn_name}→Real']['f1']:.4f} | Acc={results[f'{syn_name}→Real']['acc']:.4f}")

    # Real→Synth
    print(f"     Real→Synth...")
    results[f"Real→{syn_name}"] = evaluate(lstm_real, X_syn_te, y_syn_te)
    print(f"     F1={results[f'Real→{syn_name}']['f1']:.4f} | Acc={results[f'Real→{syn_name}']['acc']:.4f}")

    # Synth→Synth
    print(f"     Synth→Synth...")
    m2 = ThermalComfortLSTM(n_features, n_classes).to(DEVICE)
    train_lstm(m2, X_syn_tr, y_syn_tr, None, None, epochs=3, quiet=True)
    results[f"{syn_name}→{syn_name}"] = evaluate(m2, X_syn_te, y_syn_te)
    print(f"     F1={results[f'{syn_name}→{syn_name}']['f1']:.4f} | Acc={results[f'{syn_name}→{syn_name}']['acc']:.4f}")

# ============================================================
# 7. Summary
# ============================================================
print("\n" + "=" * 60)
print("[6/6] 2×2 MATRIX RESULTS")
print("=" * 60)
print(f"\n{'':>30s} | {'Test on Real':>16s} | {'Test on Synthetic':>20s}")
print(f"{'':>30s} | {'F1':>7s} {'Acc':>7s} | {'F1':>7s} {'Acc':>7s}")
print("-" * 70)

syn_names = list(synth_data.keys())
rr = results["Real→Real"]
first_syn = syn_names[0]
rs0 = results.get(f"Real→{first_syn}", {"f1":0,"acc":0})
print(f"{'Train on Real':>30s} | {rr['f1']:7.4f} {rr['acc']:7.4f} | {rs0['f1']:7.4f} {rs0['acc']:7.4f}")

for sn in syn_names:
    sr = results.get(f"{sn}→Real", {"f1":0,"acc":0})
    ss = results.get(f"{sn}→{sn}", {"f1":0,"acc":0})
    print(f"{'Train on '+sn:>30s} | {sr['f1']:7.4f} {sr['acc']:7.4f} | {ss['f1']:7.4f} {ss['acc']:7.4f}")

pd.DataFrame(results).T.to_csv("simple_2x2_results.csv")
print("\nDONE")
