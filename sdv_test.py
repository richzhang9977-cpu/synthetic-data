"""SDV-Only Test — GaussianCopula, CTGAN, TVAE on AutoTherm"""
import warnings; warnings.filterwarnings("ignore")
import time, os, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from scipy import stats
from scipy.spatial.distance import jensenshannon

# SDV
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import classification_report, f1_score, accuracy_score
import xgboost as xgb

# Datasets
from datasets import load_dataset

sns.set_style("whitegrid")

print("=" * 60)
print("SDV-ONLY TEST — AutoTherm Synthetic Data Quality")
print("=" * 60)

# ============================================================
# 1. Load & Prepare Data
# ============================================================
print("\n[1/6] Loading AutoTherm data...")
N_SAMPLES = 5000
ds = load_dataset("kopetri/AutoTherm", "combined", streaming=True, split="train")

# Balance labels
label_bins = {}
for r in ds:
    lbl = str(r["Label"])
    if lbl not in label_bins: label_bins[lbl] = []
    if len(label_bins[lbl]) < 1200:
        label_bins[lbl].append(r)
    if sum(len(v) for v in label_bins.values()) >= N_SAMPLES:
        break

rows = [r for v in label_bins.values() for r in v]
df = pd.DataFrame(rows)

# Drop IDs
for c in ["file_name"]:
    if c in df.columns: df = df.drop(columns=[c])

TARGET = "Label"
print(f"   Loaded {len(df)} rows, labels: {df[TARGET].value_counts().sort_index().to_dict()}")

# Column type detection
cat_cols, cont_cols = [], []
for col in df.columns:
    if col == TARGET: continue
    d = df[col]
    if d.dtype == object or str(d.dtype) == "string":
        cat_cols.append(col)
    elif d.nunique() <= 15:
        cat_cols.append(col)
    else:
        cont_cols.append(col)

print(f"   {len(cat_cols)} categorical, {len(cont_cols)} continuous")

# Clean
df_clean = df.copy()
for c in cat_cols: df_clean[c] = df_clean[c].fillna("missing").astype(str)
for c in cont_cols: df_clean[c] = df_clean[c].fillna(df_clean[c].median())
df_clean[TARGET] = df_clean[TARGET].astype(str)

# Split
le = LabelEncoder()
train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean[TARGET])
sdv_train = train_df[cat_cols + cont_cols + [TARGET]].copy()
n_classes = df_clean[TARGET].nunique()

print(f"   Train: {len(train_df)} | Test: {len(test_df)} | Classes: {n_classes}")

# Encode for evaluation
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train = train_df[cat_cols + cont_cols].copy()
X_test = test_df[cat_cols + cont_cols].copy()
X_train[cat_cols] = oe.fit_transform(X_train[cat_cols])
X_test[cat_cols] = oe.transform(X_test[cat_cols])
y_train_enc = le.fit_transform(train_df[TARGET].astype(str))
y_test_enc = le.transform(test_df[TARGET].astype(str))

# ============================================================
# 2. Real Data Baseline
# ============================================================
print("\n[2/6] Training real-data baseline...")
xgb_real = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                              random_state=42, eval_metric="mlogloss")
xgb_real.fit(X_train, y_train_enc)
y_pred_real = xgb_real.predict(X_test)
BASELINE_F1 = f1_score(y_test_enc, y_pred_real, average="macro")
BASELINE_ACC = accuracy_score(y_test_enc, y_pred_real)
print(f"   Real-data XGBoost: Macro-F1={BASELINE_F1:.4f}, Acc={BASELINE_ACC:.4f}")

# ============================================================
# 3. SDV Metadata
# ============================================================
print("\n[3/6] Creating SDV metadata...")
metadata = SingleTableMetadata()
metadata.detect_from_dataframe(sdv_train)
metadata.update_column(column_name=TARGET, sdtype="categorical")
for c in cat_cols: metadata.update_column(column_name=c, sdtype="categorical")
for c in cont_cols: metadata.update_column(column_name=c, sdtype="numerical")
try: metadata.remove_primary_key()
except: pass
print("   Metadata ready")

# ============================================================
# 4. Generate Synthetic Data
# ============================================================
SYNTH_ROWS = len(train_df)
results = {}
synthetic_data = {}

# --- 4a. GaussianCopula ---
print("\n[4a/6] GaussianCopula...")
t0 = time.time()
gc = GaussianCopulaSynthesizer(metadata)
gc.fit(sdv_train)
gc_synth = gc.sample(num_rows=SYNTH_ROWS)
results["GaussianCopula"] = {"time": time.time() - t0}
synthetic_data["GaussianCopula"] = gc_synth
print(f"   Done in {results['GaussianCopula']['time']:.1f}s, shape={gc_synth.shape}")

# --- 4b. CTGAN ---
print("\n[4b/6] CTGAN (300 epochs)...")
t0 = time.time()
ctgan = CTGANSynthesizer(metadata, epochs=300, cuda=True, verbose=False)
ctgan.fit(sdv_train)
ctgan_synth = ctgan.sample(num_rows=SYNTH_ROWS)
results["CTGAN"] = {"time": time.time() - t0}
synthetic_data["CTGAN"] = ctgan_synth
print(f"   Done in {results['CTGAN']['time']:.1f}s, shape={ctgan_synth.shape}")

# --- 4c. TVAE ---
print("\n[4c/6] TVAE (300 epochs)...")
t0 = time.time()
tvae = TVAESynthesizer(metadata, epochs=300, cuda=True, verbose=False)
tvae.fit(sdv_train)
tvae_synth = tvae.sample(num_rows=SYNTH_ROWS)
results["TVAE"] = {"time": time.time() - t0}
synthetic_data["TVAE"] = tvae_synth
print(f"   Done in {results['TVAE']['time']:.1f}s, shape={tvae_synth.shape}")

# ============================================================
# 5. Statistical Fidelity Evaluation
# ============================================================
print("\n[5/6] Evaluating Statistical Fidelity...\n")

def evaluate_fidelity(real, synth, cat_cols, cont_cols, target):
    m = {}
    # Wasserstein (continuous)
    wd = []
    for c in cont_cols:
        rv = pd.to_numeric(real[c], errors="coerce").dropna()
        sv = pd.to_numeric(synth[c], errors="coerce").dropna()
        if len(rv) > 0 and len(sv) > 0:
            wd.append(stats.wasserstein_distance(rv, sv))
    m["wasserstein"] = np.mean(wd) if wd else np.nan

    # KS (continuous)
    ks = []
    for c in cont_cols:
        rv = pd.to_numeric(real[c], errors="coerce").dropna()
        sv = pd.to_numeric(synth[c], errors="coerce").dropna()
        if len(rv) > 0 and len(sv) > 0:
            ks.append(stats.ks_2samp(rv, sv).statistic)
    m["ks"] = np.mean(ks) if ks else np.nan

    # JS Divergence (categorical + target)
    js = []
    for c in cat_cols + [target]:
        rc = real[c].value_counts(normalize=True)
        sc = synth[c].value_counts(normalize=True)
        all_c = list(set(rc.index) | set(sc.index))
        p = np.array([rc.get(x, 0) for x in all_c])
        q = np.array([sc.get(x, 0) for x in all_c])
        js.append(jensenshannon(p, q) if max(p.sum(), q.sum()) > 0 else 0)
    m["js_divergence"] = np.mean(js)

    # Correlation diff (continuous)
    rc = real[cont_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
    scc = synth[cont_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
    common = [c for c in rc.columns if c in scc.columns]
    if len(common) > 1:
        diff = (rc.loc[common, common] - scc.loc[common, common]).abs()
        m["corr_diff"] = diff.values[np.triu_indices_from(diff.values, k=1)].mean()
    else:
        m["corr_diff"] = np.nan

    # Target distribution diff
    rd = real[target].value_counts(normalize=True).sort_index()
    sd = synth[target].value_counts(normalize=True).sort_index()
    all_l = sorted(set(rd.index) | set(sd.index))
    m["target_diff"] = np.mean([abs(rd.get(l, 0) - sd.get(l, 0)) for l in all_l])
    return m

# Limit comparison to 4000 samples for speed
n_cmp = min(len(sdv_train), 4000)
real_sample = sdv_train.sample(n=n_cmp, random_state=42)

for name, synth in synthetic_data.items():
    synth_sample = synth.sample(n=min(len(synth), n_cmp), random_state=42)
    fid = evaluate_fidelity(real_sample, synth_sample, cat_cols, cont_cols, TARGET)
    results[name].update(fid)
    print(f"   {name:20s} | Wasserstein={fid['wasserstein']:.4f} | KS={fid['ks']:.4f} | JS={fid['js_divergence']:.4f} | CorrDiff={fid['corr_diff']:.4f} | TargetDiff={fid['target_diff']:.4f}")

# ============================================================
# 6. TSTR — ML Utility
# ============================================================
print("\n[6/6] Evaluating ML Utility (TSTR)...\n")

for name, synth in synthetic_data.items():
    # Prepare synthetic training set
    st = synth.copy()
    if TARGET not in st.columns:
        results[name]["tstr_f1"] = np.nan
        continue

    X_syn = st[cat_cols + cont_cols].copy()
    for c in cat_cols: X_syn[c] = X_syn[c].astype(str)
    X_syn[cat_cols] = oe.transform(X_syn[cat_cols])

    y_syn_raw = st[TARGET].astype(str)
    valid = y_syn_raw.isin(le.classes_)
    if valid.sum() < 100:
        results[name]["tstr_f1"] = np.nan
        continue

    X_syn = X_syn.loc[valid]
    y_syn = le.transform(y_syn_raw[valid])

    model = xgb.XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1,
                               random_state=42, eval_metric="mlogloss", verbosity=0)
    model.fit(X_syn, y_syn)
    y_pred = model.predict(X_test)
    tstr_f1 = f1_score(y_test_enc, y_pred, average="macro")
    tstr_acc = accuracy_score(y_test_enc, y_pred)
    results[name]["tstr_f1"] = tstr_f1
    results[name]["tstr_acc"] = tstr_acc
    results[name]["f1_vs_real"] = tstr_f1 / BASELINE_F1 * 100
    print(f"   {name:20s} | TSTR Macro-F1={tstr_f1:.4f} | Acc={tstr_acc:.4f} | vs Real={tstr_f1/BASELINE_F1*100:.1f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY: SDV Synthetic Data Quality on AutoTherm")
print("=" * 70)

summary = pd.DataFrame(results).T
summary.index.name = "Method"
cols_show = ["time", "wasserstein", "ks", "js_divergence", "corr_diff", "target_diff", "tstr_f1", "f1_vs_real"]
print(summary[[c for c in cols_show if c in summary.columns]].round(4).to_string())

# Rankings
print("\n--- Rankings (1=best) ---")
for metric, ascending, label in [
    ("wasserstein", True, "Wasserstein (lower)"),
    ("ks", True, "KS (lower)"),
    ("js_divergence", True, "JS (lower)"),
    ("corr_diff", True, "CorrDiff (lower)"),
    ("target_diff", True, "TargetDiff (lower)"),
    ("tstr_f1", False, "TSTR F1 (higher)"),
]:
    if metric in summary.columns:
        rank = summary[metric].rank(ascending=ascending)
        best = rank.idxmin()
        print(f"   {label:25s}: best={best}")

# Save
os.makedirs("/f/synthetic/results", exist_ok=True)
summary.to_csv("/f/synthetic/results/sdv_results.csv")
with open("/f/synthetic/results/sdv_summary.json", "w") as f:
    clean = {}
    for k, v in results.items():
        clean[k] = {kk: float(vv) if not np.isnan(vv) else None for kk, vv in v.items()}
    json.dump({"results": clean, "baseline_f1": BASELINE_F1, "baseline_acc": BASELINE_ACC}, f, indent=2)

print(f"\nResults saved to /f/synthetic/results/")
print("DONE")
