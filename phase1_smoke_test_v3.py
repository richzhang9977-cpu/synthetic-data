"""Phase 1 Smoke Test v3 — all fixes applied"""
import sys, time, warnings, os
warnings.filterwarnings("ignore")

print("=" * 60)
print("PHASE 1 SMOKE TEST v3")
print("=" * 60)

# 1. Imports
print("\n[1/7] Checking imports...")
import torch; print(f"  torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
import pandas as pd; import numpy as np
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer; print("  sdv OK")
from sdv.metadata import SingleTableMetadata; print("  sdv.metadata OK")
from datasets import load_dataset; print("  datasets OK")
import xgboost as xgb; print(f"  xgboost {xgb.__version__}")
from sklearn.model_selection import train_test_split; print("  sklearn OK")
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from be_great import GReaT; print("  be-great OK")
from realtabformer import REaLTabFormer; print("  realtabformer OK")
import peft; print(f"  peft {peft.__version__}")
print("All imports passed!")

# 2. GPU test
print("\n[2/7] Testing GPU...")
x = torch.randn(500, 500).cuda(); y = x @ x.T
print(f"GPU OK (mean={y.mean().item():.4f})")
del x, y; torch.cuda.empty_cache()

# 3. Data loading — ensure diverse labels
print("\n[3/7] Loading AutoTherm data (20K samples, label-balanced)...")
ds = load_dataset("kopetri/AutoTherm", "combined", streaming=True, split="train")

# Collect diverse labels (max N per label)
N_PER_LABEL = 1000
MAX_TOTAL = 5000
label_bins = {}
for r in ds:
    lbl = str(r["Label"])
    if lbl not in label_bins: label_bins[lbl] = []
    if len(label_bins[lbl]) < N_PER_LABEL:
        label_bins[lbl].append(r)
    if sum(len(v) for v in label_bins.values()) >= MAX_TOTAL:
        break

rows = [r for v in label_bins.values() for r in v]
df = pd.DataFrame(rows)
print(f"Loaded {len(df)} rows x {len(df.columns)} cols")
print(f"Label distribution: {df['Label'].value_counts().sort_index().to_dict()}")

# Drop IDs
for c in ["file_name"]:
    if c in df.columns: df = df.drop(columns=[c])

TARGET = "Label"

# 4. Better preprocessing
print("\n[4/7] Preprocessing...")

# Identify true categorical vs continuous
cat_cols, cont_cols = [], []
for col in df.columns:
    if col == TARGET: continue
    d = df[col]
    if d.dtype == object or str(d.dtype) == "string":
        cat_cols.append(col)
    else:
        # Numeric but might be categorical if low cardinality
        if d.nunique() <= 15:
            cat_cols.append(col)
        else:
            cont_cols.append(col)
print(f"  Categorical: {len(cat_cols)} | Continuous: {len(cont_cols)}")

df_clean = df.copy()
for col in cat_cols:
    df_clean[col] = df_clean[col].fillna("missing").astype(str)
for col in cont_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
df_clean[TARGET] = df_clean[TARGET].astype(str)

train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42, stratify=df_clean[TARGET])
sdv_train = train_df[cat_cols + cont_cols + [TARGET]].copy()
print(f"  Train: {len(sdv_train)} | Test: {len(test_df)}")

# 5. SDV methods
print("\n[5/7] Testing SDV methods...")

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(sdv_train)
metadata.update_column(column_name=TARGET, sdtype="categorical")
for col in cat_cols:
    metadata.update_column(column_name=col, sdtype="categorical")
for col in cont_cols:
    metadata.update_column(column_name=col, sdtype="numerical")
try:
    metadata.remove_primary_key()
except: pass

# 5a. GaussianCopula
t0 = time.time(); gc = GaussianCopulaSynthesizer(metadata); gc.fit(sdv_train); gc_synth = gc.sample(num_rows=500)
print(f"  GaussianCopula: {gc_synth.shape} in {time.time()-t0:.1f}s")

# 5b. CTGAN (reduced epochs)
t0 = time.time(); ctgan = CTGANSynthesizer(metadata, epochs=50, cuda=True, verbose=False); ctgan.fit(sdv_train); ctgan_synth = ctgan.sample(num_rows=500)
print(f"  CTGAN: {ctgan_synth.shape} in {time.time()-t0:.1f}s")

# 5c. TVAE (reduced epochs)
t0 = time.time(); tvae = TVAESynthesizer(metadata, epochs=50, cuda=True, verbose=False); tvae.fit(sdv_train); tvae_synth = tvae.sample(num_rows=500)
print(f"  TVAE: {tvae_synth.shape} in {time.time()-t0:.1f}s")

print("SDV methods OK")

# 6. LLM methods
print("\n[6/7] Testing LLM methods...")

# 6a. GReaT (with peft now installed)
t0 = time.time()
great_synth = None
try:
    great_train = train_df[cat_cols + cont_cols + [TARGET]].copy()
    for col in cat_cols: great_train[col] = great_train[col].astype(str)
    great_train[TARGET] = great_train[TARGET].astype(str)

    great_model = GReaT(
        llm="distilgpt2", epochs=5, batch_size=16,
        efficient_finetuning="lora", experiment_dir="/f/great_test3"
    )
    great_model.fit(great_train)
    great_synth = great_model.sample(n_samples=200)
    print(f"  GReaT: {great_synth.shape} in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  GReaT: FAILED — {str(e)[:300]}")

# 6b. REaLTabFormer (fixed API: explicit generate_kwargs)
t0 = time.time()
rtf_synth = None
try:
    rtf_train = train_df[cat_cols + cont_cols + [TARGET]].copy()
    for col in cat_cols: rtf_train[col] = rtf_train[col].astype(str)
    rtf_train[TARGET] = rtf_train[TARGET].astype(str)
    rtf_train = rtf_train.reset_index(drop=True)

    rtf_model = REaLTabFormer(
        model_type="tabular", checkpoints_dir="/f/rtf_test3",
        epochs=3, train_size=0.8
    )
    rtf_model.fit(rtf_train)
    # FIX: pass generate_kwargs explicitly
    rtf_synth = rtf_model.sample(n_samples=200, generate_kwargs={})
    print(f"  REaLTabFormer: {rtf_synth.shape} in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"  REaLTabFormer: FAILED — {str(e)[:300]}")

print("LLM check done")

# 7. Quick evaluation
print("\n[7/7] Quick evaluation...")
synth_map = {"GaussianCopula": gc_synth, "CTGAN": ctgan_synth, "TVAE": tvae_synth}
if great_synth is not None: synth_map["GReaT"] = great_synth
if rtf_synth is not None: synth_map["REaLTabFormer"] = rtf_synth

for name, synth in synth_map.items():
    common = [c for c in sdv_train.columns if c in synth.columns]
    target_present = TARGET in synth.columns
    n_common = len(common)
    print(f"  {name}: {n_common}/{len(sdv_train.columns)} cols, Target={target_present}")

    if target_present:
        real_dist = sdv_train[TARGET].value_counts(normalize=True).sort_index()
        synth_dist = synth[TARGET].value_counts(normalize=True).sort_index()
        print(f"    Real dist: {dict(real_dist)}")
        print(f"    Synth dist: {dict(synth_dist)}")

print("\n" + "=" * 60)
print("SMOKE TEST v3 COMPLETE!")
print("=" * 60)
