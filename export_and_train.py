"""Paper-correct 2x2 matrix: paper's dataloader + our LSTM (matching paper architecture)

Uses paper's TC_Dataloader for proper per-participant split,
then our verified LSTM implementation for training/evaluation.
"""
import warnings; warnings.filterwarnings("ignore")
import json, time, os, re, sys
import numpy as np; import pandas as pd
import torch; import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

sys.path.insert(0, "F:/thermal_paper")

print("=" * 60)
print("2x2 MATRIX — Paper Dataloader + Verified LSTM")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Config
DATA_DIR = "F:/thermal_data"; CSV_DIR = f"{DATA_DIR}/csv"
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs("F:/thermal_paper/dataloaders/splits", exist_ok=True)

# ============================================================
# Paper's LSTM Architecture (exact from learning_models.py)
# ============================================================
class ThermalComfortLSTM(nn.Module):
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

def train_lstm(model, loader, val_loader, epochs=1):
    """Train with paper's exact hyperparams: Adam(1e-4), MSE, LambdaLR"""
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: np.power(0.9999999, s))
    crit = nn.MSELoss()
    amp = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None

    for ep in range(epochs):
        model.train(); total_loss = 0
        for bx, by in loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE); opt.zero_grad()
            if amp:
                with torch.amp.autocast("cuda"):
                    pred = model(bx)
                    loss = crit(pred, nn.functional.one_hot(by.long(), num_classes).float())
                amp.scale(loss).backward(); amp.step(opt); amp.update()
            else:
                pred = model(bx); loss = crit(pred, nn.functional.one_hot(by.long(), num_classes).float())
                loss.backward(); opt.step()
            sched.step(); total_loss += loss.item()

        if val_loader:
            model.eval(); all_pred, all_true = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    pred = model(bx.to(DEVICE)).argmax(-1).cpu()
                    all_pred.extend(pred.numpy()); all_true.extend(by.numpy())
            acc = accuracy_score(all_true, all_pred)
            f1 = f1_score(all_true, all_pred, average="macro", zero_division=0)
            print(f"   Epoch {ep+1}: Loss={total_loss/len(loader):.4f} | Val Acc={acc:.4f} | Val F1={f1:.4f}")

def evaluate(model, loader):
    model.eval(); all_pred, all_true = [], []
    with torch.no_grad():
        for bx, by in loader:
            pred = model(bx.to(DEVICE)).argmax(-1).cpu()
            all_pred.extend(pred.numpy()); all_true.extend(by.numpy())
    return {
        "acc": accuracy_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred, average="macro", zero_division=0)
    }

# ============================================================
# STEP 1: Export indoor_frames → CSV files (one per session)
# ============================================================
print("\n[1/5] Exporting data...")
existing = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
if len(existing) >= 5:
    all_sessions = sorted([f.replace('.csv','') for f in existing])
    print(f"   Found {len(all_sessions)} existing CSV files, skipping export.")
else:
    from datasets import load_dataset
    ds = load_dataset("kopetri/AutoTherm", "indoor_frames", streaming=True, split="train")
    sessions = {}
    for i, sample in enumerate(ds):
        if i % 2000 == 0: print(f"   ... {i} frames, {len(sessions)} sessions")
        if i >= 20000: break
        try:
            meta = json.loads(sample["jpg.json"].decode("utf-8")
                           if isinstance(sample["jpg.json"],bytes) else sample["jpg.json"])
        except: continue
        sess = meta.get("file_name","").split("/")[0]
        sessions.setdefault(sess, []).append(meta)

    HEADER = ["Timestamp","Age","Gender","Weight","Height","Bodyfat","Bodytemp",
              "Sport-Last-Hour","Time-Since-Meal","Tiredness","Clothing-Level",
              "Radiation-Temp","PCE-Ambient-Temp","Air-Velocity","Metabolic-Rate",
              "Emotion-Self","Emotion-ML","RGB_Frontal_View",
              "Nose","Neck","RShoulder","RElbow","LShoulder","LElbow",
              "REye","LEye","REar","LEar",
              "Wrist_Skin_Temperature","Heart_Rate","GSR",
              "Ambient_Temperature","Ambient_Humidity","Label"]

    all_sessions = []
    for sess, frames in sessions.items():
        frames.sort(key=lambda f: int(re.search(r'frame_f_(\d+)',
            str(f.get("file_name",""))).group(1) if re.search(r'frame_f_(\d+)',
            str(f.get("file_name",""))) else 0))
        rows = [";".join([str(f.get(h, "")) for h in HEADER]) for f in frames]
        with open(f"{CSV_DIR}/{sess}.csv", "w") as f: f.write("\n".join(rows))
        all_sessions.append(sess)
    print(f"   Exported {len(all_sessions)} sessions")

# ============================================================
# STEP 2: Create participant-level splits + paper dataloaders
# ============================================================
print("\n[2/5] Creating splits & dataloaders...")

np.random.seed(42); shuffled = sorted(all_sessions); np.random.shuffle(shuffled)
n = len(shuffled)
train_s = shuffled[:int(n*0.7)]; val_s = shuffled[int(n*0.7):int(n*0.85)]; test_s = shuffled[int(n*0.85):]

for name, sessions in [("training",train_s),("validation",val_s),("test",test_s)]:
    with open(f"F:/thermal_paper/dataloaders/splits/{name}_60_real.txt","w") as f:
        f.write("\n".join([f"{s}.csv" for s in sessions]))
    print(f"   {name}: {len(sessions)} sessions")

with open("F:/thermal_paper/dataloaders/splits/real_all.txt","w") as f:
    f.write("\n".join([f"{s}.csv" for s in all_sessions]))

# Use paper's dataloader
from dataloaders.tc_dataloader import TC_Dataloader
PAPER_COLS = [11, 28, 31, 32, 33]  # Radiation-Temp, Wrist-Temp, Ambient-Temp, Humidity, Label
SEQ = 10; SCALE = 7

print("   Creating dataloaders...")
train_ds = TC_Dataloader(root=f"{CSV_DIR}/", split="training", scale=SCALE, cols=PAPER_COLS,
                          use_sequence=True, sequence_size=SEQ, preprocess=True)
val_ds = TC_Dataloader(root=f"{CSV_DIR}/", split="validation", scale=SCALE, cols=PAPER_COLS,
                        use_sequence=True, sequence_size=SEQ, preprocess=True)
test_ds = TC_Dataloader(root=f"{CSV_DIR}/", split="test", scale=SCALE, cols=PAPER_COLS,
                         use_sequence=True, sequence_size=SEQ, preprocess=True)

# Wrap in DataLoaders
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

n_features = len(train_ds.columns) - 1
num_classes = SCALE
print(f"   Features: {n_features} | Classes: {num_classes} | Train batches: {len(train_loader)}")

# ============================================================
# STEP 3: Real→Real (Baseline)
# ============================================================
print("\n[3/5] REAL→REAL Baseline...")
lstm_real = ThermalComfortLSTM(n_features, num_classes).to(DEVICE)
train_lstm(lstm_real, train_loader, val_loader, epochs=2)
real_real = evaluate(lstm_real, test_loader)
print(f"   Real→Real: Acc={real_real['acc']:.4f} | F1={real_real['f1']:.4f}")
torch.save(lstm_real.state_dict(), f"{DATA_DIR}/lstm_real.pt")

# ============================================================
# STEP 4: Generate SDV synthetic data (from training set only!)
# ============================================================
print("\n[4/5] SDV synthesis...")
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# Collect all training frames as flat table for SDV
train_frames = pd.concat([pd.read_csv(f"{CSV_DIR}/{s}.csv", delimiter=";") for s in train_s])
train_frames = train_frames[['Radiation-Temp','Wrist_Skin_Temperature','Ambient_Temperature','Ambient_Humidity','Label']]
train_frames['Label'] = train_frames['Label'].fillna(0).astype(int).astype(str)
for c in ['Radiation-Temp','Wrist_Skin_Temperature','Ambient_Temperature','Ambient_Humidity']:
    train_frames[c] = pd.to_numeric(train_frames[c], errors='coerce').fillna(0)

# Also for test: collect real test frames for Real→Synth eval
test_frames = pd.concat([pd.read_csv(f"{CSV_DIR}/{s}.csv", delimiter=";") for s in test_s])
test_frames = test_frames[['Radiation-Temp','Wrist_Skin_Temperature','Ambient_Temperature','Ambient_Humidity','Label']]
test_frames['Label'] = test_frames['Label'].fillna(0).astype(int).astype(str)
for c in ['Radiation-Temp','Wrist_Skin_Temperature','Ambient_Temperature','Ambient_Humidity']:
    test_frames[c] = pd.to_numeric(test_frames[c], errors='coerce').fillna(0)

meta = SingleTableMetadata(); meta.detect_from_dataframe(train_frames)
meta.update_column(column_name='Label', sdtype='categorical')
for c in ['Radiation-Temp','Wrist_Skin_Temperature','Ambient_Temperature','Ambient_Humidity']:
    meta.update_column(column_name=c, sdtype='numerical')
try: meta.remove_primary_key()
except: pass

# Use same preprocessing as paper (from TC_Dataloader)
from sklearn.preprocessing import StandardScaler
feat_cols = ['Radiation-Temp','Wrist_Skin_Temperature','Ambient_Temperature','Ambient_Humidity']
scaler = StandardScaler()
train_X_scaled = scaler.fit_transform(train_frames[feat_cols].astype(float))
test_X_scaled = scaler.transform(test_frames[feat_cols].astype(float))

synth_data = {}
for name, cls, ep in [("GaussianCopula", GaussianCopulaSynthesizer, 0),
                       ("CTGAN", CTGANSynthesizer, 300),
                       ("TVAE", TVAESynthesizer, 300)]:
    cache = f"{DATA_DIR}/synth_{name}.csv"
    if os.path.exists(cache):
        df_s = pd.read_csv(cache); print(f"   {name}: loaded ({df_s.shape})")
    else:
        print(f"   {name}: generating..."); t0 = time.time()
        s = cls(meta, epochs=ep, cuda=True, verbose=False) if ep>0 else cls(meta)
        s.fit(train_frames); df_s = s.sample(num_rows=len(train_frames))
        df_s.to_csv(cache, index=False)
        print(f"   {name}: done in {time.time()-t0:.1f}s ({df_s.shape})")
    synth_data[name] = df_s

# ============================================================
# STEP 5: 2x2 Matrix
# ============================================================
print("\n[5/5] 2x2 MATRIX...")

def build_seq_loader(df, scaler, seq_len, batch_size=4):
    """Convert flat synthetic DataFrame to LSTM sequence DataLoader"""
    X = scaler.transform(df[feat_cols].astype(float))
    y = np.clip(df['Label'].astype(int).values, 0, num_classes-1)

    n_seqs = (len(y) // seq_len) - 1
    if n_seqs < 5: return None
    X_seq = X[:n_seqs*seq_len].reshape(n_seqs, seq_len, -1)
    y_seq = y[seq_len-1::seq_len][:n_seqs]
    return DataLoader(list(zip(
        torch.FloatTensor(X_seq),
        torch.LongTensor(y_seq)
    )), batch_size=batch_size, shuffle=False)

# Build real test sequences
X_real_test = test_X_scaled
y_real_test = np.clip(test_frames['Label'].astype(int).values, 0, num_classes-1)
n_seqs = (len(y_real_test) // SEQ) - 1
if n_seqs > 0:
    X_rt = torch.FloatTensor(X_real_test[:n_seqs*SEQ].reshape(n_seqs, SEQ, -1))
    y_rt = torch.LongTensor(y_real_test[SEQ-1::SEQ][:n_seqs])
    real_test_loader = DataLoader(list(zip(X_rt, y_rt)), batch_size=4, shuffle=False)
else:
    real_test_loader = test_loader  # fallback

results = {"Real→Real": real_real}

for syn_name, syn_df in synth_data.items():
    print(f"\n   --- {syn_name} ---")

    # Build synthetic sequence loader (independent from real)
    syn_loader = build_seq_loader(syn_df, scaler, SEQ)
    if syn_loader is None:
        print(f"     SKIP: too few sequences")
        results[f"{syn_name}→Real"] = {"acc": 0, "f1": 0}
        results[f"Real→{syn_name}"] = {"acc": 0, "f1": 0}
        results[f"{syn_name}→{syn_name}"] = {"acc": 0, "f1": 0}
        continue

    # Split synthetic into train/test for Synth→Synth
    syn_total = len(syn_loader.dataset)
    syn_n_train = int(syn_total * 0.7)
    syn_train_subset = torch.utils.data.Subset(syn_loader.dataset, range(syn_n_train))
    syn_test_subset = torch.utils.data.Subset(syn_loader.dataset, range(syn_n_train, syn_total))
    syn_train_loader = DataLoader(syn_train_subset, batch_size=4, shuffle=True)
    syn_test_loader = DataLoader(syn_test_subset, batch_size=4, shuffle=False)

    # Synth→Real
    print("     Synth→Real...")
    m = ThermalComfortLSTM(n_features, num_classes).to(DEVICE)
    train_lstm(m, syn_train_loader, None, epochs=1)
    results[f"{syn_name}→Real"] = evaluate(m, real_test_loader)
    print(f"     Synth→Real: F1={results[f'{syn_name}→Real']['f1']:.4f} | Acc={results[f'{syn_name}→Real']['acc']:.4f}")

    # Real→Synth (use saved real model, test on synthetic)
    print("     Real→Synth...")
    lm = ThermalComfortLSTM(n_features, num_classes).to(DEVICE)
    lm.load_state_dict(torch.load(f"{DATA_DIR}/lstm_real.pt"))
    results[f"Real→{syn_name}"] = evaluate(lm, syn_test_loader)
    print(f"     Real→Synth: F1={results[f'Real→{syn_name}']['f1']:.4f} | Acc={results[f'Real→{syn_name}']['acc']:.4f}")

    # Synth→Synth
    print("     Synth→Synth...")
    m2 = ThermalComfortLSTM(n_features, num_classes).to(DEVICE)
    train_lstm(m2, syn_train_loader, None, epochs=1)
    results[f"{syn_name}→{syn_name}"] = evaluate(m2, syn_test_loader)
    print(f"     Synth→Synth: F1={results[f'{syn_name}→{syn_name}']['f1']:.4f} | Acc={results[f'{syn_name}→{syn_name}']['acc']:.4f}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("2x2 MATRIX RESULTS")
print("=" * 60)
print(f"\n{'':25s} | {'Test on Real':>15s} | {'Test on Synthetic':>20s}")
print(f"{'':25s} | {'F1':>6s} {'Acc':>7s} | {'F1':>6s} {'Acc':>7s}")
print("-" * 65)
print(f"{'Train on Real':25s} | {real_real['f1']:6.4f} {real_real['acc']:7.4f}", end="")
for syn_name in synth_data:
    rs = results.get(f"Real→{syn_name}", {"f1":0,"acc":0})
    if syn_name == list(synth_data.keys())[0]:
        print(f" | {rs['f1']:6.4f} {rs['acc']:7.4f} ({syn_name})")

for syn_name in synth_data:
    sr = results.get(f"{syn_name}→Real", {"f1":0,"acc":0})
    ss = results.get(f"{syn_name}→{syn_name}", {"f1":0,"acc":0})
    print(f"{'Train on '+syn_name:25s} | {sr['f1']:6.4f} {sr['acc']:7.4f} | {ss['f1']:6.4f} {ss['acc']:7.4f}")

os.makedirs("F:/synthetic/results", exist_ok=True)
pd.DataFrame(results).T.to_csv("F:/synthetic/results/2x2_matrix.csv")
print("\nDONE")
