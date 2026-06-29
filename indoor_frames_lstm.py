"""LSTM on indoor_frames with paper's methodology (Section 7.7)
- Paper features: Radiation-Temp, Ambient_Humidity, Ambient_Temperature, Wrist_Skin_Temperature
- Paper architecture: LSTM(hidden=128, layers=2, dropout=0.5) → FC → Sigmoid
- Paper training: MSE loss, Adam(lr=1e-4), LambdaLR scheduler
- Temporal sequences from indoor_frames (grouped by recording session)
"""
import warnings; warnings.filterwarnings("ignore")
import json, time, os, re
import numpy as np; import pandas as pd
import torch; import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report
from datasets import load_dataset

print("=" * 60)
print("AutoTherm Paper Reproduction: indoor_frames + LSTM")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# 1. Paper's LSTM Architecture (Section 7.7)
# ============================================================
class ThermalComfortLSTM(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=128, n_layers=2, dropout=0.5):
        super().__init__()
        self.n_layers = n_layers
        self.lstm = nn.LSTM(num_features, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
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

# ============================================================
# 2. Load indoor_frames & parse tabular data from jpg.json
# ============================================================
print("\n[1/5] Loading indoor_frames...")
PAPER_FEATURES = ["Radiation-Temp", "Ambient_Humidity", "Ambient_Temperature", "Wrist_Skin_Temperature"]
TARGET = "Label"
SEQUENCE_WINDOW = 10
MAX_SAMPLES = 30000

ds = load_dataset("kopetri/AutoTherm", "indoor_frames", streaming=True, split="train")

sessions = {}
for i, sample in enumerate(ds):
    if i >= MAX_SAMPLES: break
    try:
        meta = json.loads(sample["jpg.json"].decode("utf-8") if isinstance(sample["jpg.json"], bytes) else sample["jpg.json"])
    except: continue

    fn = meta.get("file_name", "")
    session = fn.split("/")[0] if "/" in fn else fn
    row = {f: meta.get(f) for f in PAPER_FEATURES + [TARGET]}
    row["_session"] = session
    sessions.setdefault(session, []).append(row)

# Filter sessions with enough frames + sort by frame number
valid = {k: v for k, v in sessions.items() if len(v) >= SEQUENCE_WINDOW}
for sess in valid:
    valid[sess].sort(key=lambda f: int(re.search(r'frame_f_(\d+)', str(f.get("_session", ""))).group(1)
                                       if re.search(r'frame_f_(\d+)', str(f.get("_session", ""))) else 0))

all_rows = [f for v in valid.values() for f in v]
df_all = pd.DataFrame(all_rows)
print(f"   {len(valid)} sessions | {len(df_all)} frames | {len(PAPER_FEATURES)} features")

# ============================================================
# 3. Preprocess
# ============================================================
print("\n[2/5] Preprocessing...")
df_all[TARGET] = df_all[TARGET].astype(int).astype(str)

le = LabelEncoder(); le.fit(sorted(df_all[TARGET].unique(), key=int))
df_all["_y"] = le.transform(df_all[TARGET])

scaler = StandardScaler()
df_all[PAPER_FEATURES] = scaler.fit_transform(df_all[PAPER_FEATURES].fillna(0))

n_classes = len(le.classes_)
print(f"   Classes: {n_classes} | Labels: {list(le.classes_)}")
vc = df_all[TARGET].value_counts().sort_index()
print(f"   Distribution: {dict(zip(vc.index, vc.values))}")

# ============================================================
# 4. Split by session + build temporal sequences
# ============================================================
print("\n[3/5] Splitting by session & building sequences...")
sess_ids = sorted(valid.keys())
np.random.seed(42); np.random.shuffle(sess_ids)
n_tr = max(1, int(len(sess_ids) * 0.7))

train_sess = set(sess_ids[:n_tr])
test_sess = set(sess_ids[n_tr:])
print(f"   Train sessions: {len(train_sess)} | Test sessions: {len(test_sess)}")

def build_sequences(df, session_set):
    X_list, y_list = [], []
    for sess in session_set:
        frames = df[df["_session"] == sess].reset_index(drop=True)
        if len(frames) < SEQUENCE_WINDOW: continue
        vals = frames[PAPER_FEATURES].values.astype(np.float32)
        labels = frames["_y"].values
        for i in range(len(vals) - SEQUENCE_WINDOW + 1):
            X_list.append(vals[i:i+SEQUENCE_WINDOW])
            y_list.append(labels[i+SEQUENCE_WINDOW-1])
    if not X_list:
        return np.array([]).reshape(0, 1, 1), np.array([])
    return np.array(X_list), np.array(y_list)

X_train, y_train = build_sequences(df_all, train_sess)
X_test, y_test = build_sequences(df_all, test_sess)

# If test is empty (single session), use intra-session split
if len(X_test) == 0 or len(np.unique(y_test)) < 2:
    print("   Fallback to intra-session split...")
    X_all, y_all = build_sequences(df_all, set(sess_ids))
    split_pt = int(len(X_all) * 0.7)
    X_train, y_train = X_all[:split_pt], y_all[:split_pt]
    X_test, y_test = X_all[split_pt:], y_all[split_pt:]

print(f"   Train: {X_train.shape} | Test: {X_test.shape}")
print(f"   Test labels: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# ============================================================
# 5. Train LSTM on REAL data + TSTR on SDV synthetic data
# ============================================================
X_train_t = torch.FloatTensor(X_train); X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

def train_lstm(X_tr, y_tr, X_val, y_val, epochs=100, quiet=False):
    """Train LSTM with paper's exact hyperparameters"""
    model = ThermalComfortLSTM(len(PAPER_FEATURES), n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=4, shuffle=True)
    amp = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for ep in range(epochs):
        model.train()
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
            sched.step()

        if not quiet and (ep+1) % 25 == 0:
            model.eval()
            with torch.no_grad():
                vp = model(X_val.to(DEVICE)).argmax(-1).cpu()
                print(f"   Ep {ep+1:3d} | Acc={(vp==y_val).float().mean():.4f} | F1={f1_score(y_val.numpy(), vp, average='macro'):.4f}")
    model.eval(); return model

# --- Train on REAL data ---
print("\n[4/5] Training LSTM on REAL indoor_frames...")
lstm_real = train_lstm(X_train_t, torch.FloatTensor(y_train), X_test_t, y_test_t)
with torch.no_grad():
    y_pred_real = lstm_real(X_test_t.to(DEVICE)).argmax(-1).cpu().numpy()

REAL_F1 = f1_score(y_test, y_pred_real, average="macro")
REAL_ACC = accuracy_score(y_test, y_pred_real)
print(f"\n   *** REAL LSTM (paper features, indoor_frames, temporal sequences) ***")
print(f"   Macro-F1: {REAL_F1:.4f} | Accuracy: {REAL_ACC:.4f}")
print(f"   Paper baseline: LSTM=59.1% (7-class, vehicle, per-participant CV)\n")
print(classification_report(y_test, y_pred_real, zero_division=0))

# --- Generate SDV synthetic data ---
print("[5/5] SDV synthesis + LSTM TSTR...")

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Flatten temporal data for SDV (SDV generates static rows, not sequences)
flat_df = df_all[PAPER_FEATURES + [TARGET]].copy()
flat_df[TARGET] = flat_df[TARGET].astype(str)

meta = SingleTableMetadata(); meta.detect_from_dataframe(flat_df)
meta.update_column(column_name=TARGET, sdtype="categorical")
for c in PAPER_FEATURES: meta.update_column(column_name=c, sdtype="numerical")
try: meta.remove_primary_key()
except: pass

SYNTH_ROWS = len(flat_df)
N_SYNTH_SEQ = len(X_train)  # Match number of training sequences

tstr_results = {}
for name, cls, ep in [("GaussianCopula", GaussianCopulaSynthesizer, 0),
                       ("CTGAN", CTGANSynthesizer, 300),
                       ("TVAE", TVAESynthesizer, 300)]:
    cache = f"F:/synthetic/synth_indoor_paper_{name}.csv"
    if os.path.exists(cache):
        df_s = pd.read_csv(cache)
        print(f"   {name}: loaded cached ({df_s.shape})")
    else:
        print(f"   {name}: generating..."); t0 = time.time()
        s = cls(meta, epochs=ep, cuda=True, verbose=False) if ep > 0 else cls(meta)
        s.fit(flat_df); df_s = s.sample(num_rows=SYNTH_ROWS)
        df_s.to_csv(cache, index=False)
        print(f"   {name}: done in {time.time()-t0:.1f}s ({df_s.shape})")

    # Build sequences from synthetic data for LSTM
    df_s[TARGET] = df_s[TARGET].astype(str)
    valid_mask = df_s[TARGET].isin(le.classes_)
    syn_vals = scaler.transform(df_s[PAPER_FEATURES].fillna(0))[valid_mask.values]
    syn_y = le.transform(df_s[TARGET][valid_mask.values])

    N = min(len(syn_y), N_SYNTH_SEQ * SEQUENCE_WINDOW)
    n_seqs = N // SEQUENCE_WINDOW
    if n_seqs < 10: continue
    syn_X = syn_vals[:n_seqs*SEQUENCE_WINDOW].reshape(n_seqs, SEQUENCE_WINDOW, -1)
    syn_y_seq = syn_y[SEQUENCE_WINDOW-1::SEQUENCE_WINDOW][:n_seqs]

    print(f"   {name}: {n_seqs} sequences → training LSTM..."); t0 = time.time()
    lstm_s = train_lstm(torch.FloatTensor(syn_X), torch.FloatTensor(syn_y_seq),
                        X_test_t, y_test_t, epochs=50, quiet=True)
    with torch.no_grad():
        yp = lstm_s(X_test_t.to(DEVICE)).argmax(-1).cpu().numpy()
    tstr_f1 = f1_score(y_test, yp, average="macro", zero_division=0)
    tstr_acc = accuracy_score(y_test, yp)
    tstr_results[name] = {"f1": tstr_f1, "acc": tstr_acc, "time": time.time()-t0}
    print(f"   {name}: TSTR F1={tstr_f1:.4f} | Acc={tstr_acc:.4f} | vs Real={tstr_f1/REAL_F1*100:.0f}% | {tstr_results[name]['time']:.0f}s")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS: indoor_frames (paper features) + LSTM")
print("=" * 60)
print(f"   Paper: LSTM=59.1% (7-class, vehicle, per-participant CV)")
print(f"   Ours:  REAL LSTM F1={REAL_F1:.4f} | Acc={REAL_ACC:.4f}")
print(f"\n   {'Method':20s} | {'TSTR F1':>9s} | {'TSTR Acc':>9s} | {'vs Real F1':>11s} | {'vs Paper 59.1%':>14s}")
print(f"   {'-'*75}")
for n, r in tstr_results.items():
    print(f"   {n:20s} | {r['f1']:9.4f} | {r['acc']:9.4f} | {r['f1']/REAL_F1*100:10.1f}% | {r['f1']/0.591*100:13.1f}%")

os.makedirs("F:/synthetic/results", exist_ok=True)
pd.DataFrame(tstr_results).T.to_csv("F:/synthetic/results/indoor_paper_tstr.csv")
print("\nDONE")
