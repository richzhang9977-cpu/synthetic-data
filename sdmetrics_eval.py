"""Evaluate pre-generated synthetic data using SDMetrics (SDGym's underlying engine)"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd; import numpy as np; import json; import time
from tqdm.auto import tqdm

print("=" * 60)
print("SDMetrics Evaluation — AutoTherm Synthetic Data")
print("=" * 60)

# ============================================================
# 1. Load real data & synthetic data
# ============================================================
print("\n[1/5] Loading data...")

from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Load real data (same as sdv_test.py)
N_SAMPLES = 5000
ds = load_dataset("kopetri/AutoTherm", "combined", streaming=True, split="train")
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
for c in ["file_name"]:
    if c in df.columns: df = df.drop(columns=[c])

TARGET = "Label"
cat_cols, cont_cols = [], []
for col in df.columns:
    if col == TARGET: continue
    d = df[col]
    if d.dtype == object or str(d.dtype) == "string":
        cat_cols.append(col)
    elif d.nunique() <= 15: cat_cols.append(col)
    else: cont_cols.append(col)

df[TARGET] = df[TARGET].astype(str)
for c in cat_cols: df[c] = df[c].fillna("missing").astype(str)
for c in cont_cols: df[c] = df[c].fillna(df[c].median())

real_train, real_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df[TARGET])
print(f"   Real train: {len(real_train)} | Real test: {len(real_test)}")

# Load pre-generated synthetic data
synth_files = {
    "GaussianCopula": "F:/synthetic/synth_gc.csv",
    "CTGAN": "F:/synthetic/synth_ctgan.csv",
    "TVAE": "F:/synthetic/synth_tvae.csv",
}

# Check if files exist; if not, generate them from the sdv_results run
import os
synthetic_data = {}
needs_generation = False
for name, path in synth_files.items():
    if os.path.exists(path):
        synthetic_data[name] = pd.read_csv(path)
        print(f"   Loaded {name}: {synthetic_data[name].shape}")
    else:
        needs_generation = True
        print(f"   {name} file not found, will regenerate...")

if needs_generation:
    print("\n   Regenerating synthetic data...")
    from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
    from sdv.metadata import SingleTableMetadata

    sdv_train = real_train.copy()

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(sdv_train)
    metadata.update_column(column_name=TARGET, sdtype="categorical")
    for c in cat_cols: metadata.update_column(column_name=c, sdtype="categorical")
    for c in cont_cols: metadata.update_column(column_name=c, sdtype="numerical")
    try: metadata.remove_primary_key()
    except: pass

    SYNTH_ROWS = len(real_train)

    # GaussianCopula
    if "GaussianCopula" not in synthetic_data:
        print("   Generating GaussianCopula...")
        gc = GaussianCopulaSynthesizer(metadata); gc.fit(sdv_train)
        gc_s = gc.sample(num_rows=SYNTH_ROWS)
        gc_s.to_csv("F:/synthetic/synth_gc.csv", index=False)
        synthetic_data["GaussianCopula"] = gc_s

    # CTGAN
    if "CTGAN" not in synthetic_data:
        print("   Generating CTGAN...")
        ct = CTGANSynthesizer(metadata, epochs=300, cuda=True, verbose=False)
        ct.fit(sdv_train)
        ct_s = ct.sample(num_rows=SYNTH_ROWS)
        ct_s.to_csv("F:/synthetic/synth_ctgan.csv", index=False)
        synthetic_data["CTGAN"] = ct_s

    # TVAE
    if "TVAE" not in synthetic_data:
        print("   Generating TVAE...")
        tv = TVAESynthesizer(metadata, epochs=300, cuda=True, verbose=False)
        tv.fit(sdv_train)
        tv_s = tv.sample(num_rows=SYNTH_ROWS)
        tv_s.to_csv("F:/synthetic/synth_tvae.csv", index=False)
        synthetic_data["TVAE"] = tv_s

# ============================================================
# 2. SDMetrics — Statistical Metrics
# ============================================================
print("\n[2/5] SDMetrics Statistical Evaluation...")

from sdmetrics.single_table import (
    # Column shapes
    KSComplement, TVComplement,
    # Column pair trends
    CorrelationSimilarity, ContingencySimilarity,
    # New row synthesis
    NewRowSynthesis,
    # Boundary
    BoundaryAdherence,
    # ML Efficacy
    BinaryMLPClassifier, MulticlassMLPClassifier,
    BinaryDecisionTreeClassifier, MulticlassDecisionTreeClassifier,
)

# Build metadata for SDMetrics
sdmetrics_metadata = {
    "columns": {}
}
for c in cat_cols:
    sdmetrics_metadata["columns"][c] = {"sdtype": "categorical"}
for c in cont_cols:
    sdmetrics_metadata["columns"][c] = {"sdtype": "numerical"}
sdmetrics_metadata["columns"][TARGET] = {"sdtype": "categorical"}
sdmetrics_metadata["primary_key"] = None

all_results = {}

for name, synth_df in tqdm(synthetic_data.items(), desc="Evaluating"):
    # Align columns
    common_cols = [c for c in real_train.columns if c in synth_df.columns]
    real_sub = real_train[common_cols].reset_index(drop=True)
    synth_sub = synth_df[common_cols].reset_index(drop=True)
    # Ensure matching dtypes
    for c in cont_cols:
        if c in synth_sub.columns:
            synth_sub[c] = pd.to_numeric(synth_sub[c], errors="coerce")
            real_sub[c] = pd.to_numeric(real_sub[c], errors="coerce")

    # Filter metadata to common columns
    col_meta = {c: sdmetrics_metadata["columns"][c] for c in common_cols if c in sdmetrics_metadata["columns"]}

    results = {}
    n_real = min(len(real_sub), 3000)
    n_synth = min(len(synth_sub), 3000)
    real_s = real_sub.sample(n=n_real, random_state=42)
    synth_s = synth_sub.sample(n=n_synth, random_state=42)

    # Column Shapes
    try:
        ks = KSComplement.compute(real_s, synth_s)
        results["KSComplement"] = ks
    except Exception as e: results["KSComplement"] = f"ERR: {e}"

    try:
        tv = TVComplement.compute(real_s, synth_s)
        results["TVComplement"] = tv
    except Exception as e: results["TVComplement"] = f"ERR: {e}"

    # Correlation Similarity (continuous-continuous pairs)
    cont_in_both = [c for c in cont_cols if c in common_cols]
    if len(cont_in_both) >= 2:
        try:
            cs = CorrelationSimilarity.compute(real_s, synth_s)
            results["CorrelationSimilarity"] = cs
        except Exception as e: results["CorrelationSimilarity"] = f"ERR: {e}"

    # Contingency Similarity (categorical-categorical pairs)
    cat_in_both = [c for c in cat_cols if c in common_cols and c != TARGET]
    if len(cat_in_both) >= 2:
        try:
            ctg = ContingencySimilarity.compute(real_s, synth_s)
            results["ContingencySimilarity"] = ctg
        except Exception as e: results["ContingencySimilarity"] = f"ERR: {e}"

    # Boundary Adherence
    try:
        ba = BoundaryAdherence.compute(real_s, synth_s)
        results["BoundaryAdherence"] = ba
    except Exception as e: results["BoundaryAdherence"] = f"ERR: {e}"

    # New Row Synthesis (how many synthetic rows are direct copies?)
    try:
        nrs = NewRowSynthesis.compute(real_s, synth_s)
        results.update(nrs)
    except Exception as e: results["NewRowSynthesis"] = f"ERR: {e}"

    all_results[name] = results

# ============================================================
# 3. SDMetrics — ML Efficacy (Detection-based)
# ============================================================
print("\n[3/5] SDMetrics ML Efficacy (Can classifiers distinguish real vs synthetic?)...")

for name, synth_df in tqdm(synthetic_data.items(), desc="ML Efficacy"):
    common_cols = [c for c in real_train.columns if c in synth_df.columns]
    real_sub = real_train[common_cols].reset_index(drop=True)
    synth_sub = synth_df[common_cols].reset_index(drop=True)
    for c in cont_cols:
        if c in synth_sub.columns:
            synth_sub[c] = pd.to_numeric(synth_sub[c], errors="coerce")
            real_sub[c] = pd.to_numeric(real_sub[c], errors="coerce")
    for c in cat_cols:
        if c in synth_sub.columns:
            synth_sub[c] = synth_sub[c].astype(str)
            real_sub[c] = real_sub[c].astype(str)

    n = min(len(real_sub), len(synth_sub), 2000)
    real_s = real_sub.sample(n=n, random_state=42)
    synth_s = synth_sub.sample(n=n, random_state=42)

    # Binary detection: can we tell real from synthetic? (lower = more realistic)
    try:
        from sdmetrics.single_table import BinaryDecisionTreeClassifier as BinDT
        bdt = BinDT.compute(real_s, synth_s)
        all_results[name]["BinaryDT_Detection"] = bdt
    except Exception as e: all_results[name]["BinaryDT_Detection"] = f"ERR: {e}"

# ============================================================
# 4. Summary Table
# ============================================================
print("\n[4/5] Building summary...")

# Flatten for display
print("\n" + "=" * 80)
print("SDMETRICS FULL EVALUATION RESULTS")
print("=" * 80)

for name, res in all_results.items():
    print(f"\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    for metric, value in res.items():
        if isinstance(value, dict):
            print(f"  {metric}:")
            for k, v in value.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")
        elif isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")

# ============================================================
# 5. Scorecard & Recommendations
# ============================================================
print("\n\n[5/5] Scorecard")

scorecard = []
for name, res in all_results.items():
    score = {}
    score["Method"] = name

    # Statistical fidelity (0-1, higher better)
    ks_val = res.get("KSComplement", 0) if isinstance(res.get("KSComplement"), float) else 0
    tv_val = res.get("TVComplement", 0) if isinstance(res.get("TVComplement"), float) else 0
    corr_val = res.get("CorrelationSimilarity", 0) if isinstance(res.get("CorrelationSimilarity"), float) else 0
    bound_val = res.get("BoundaryAdherence", 0) if isinstance(res.get("BoundaryAdherence"), float) else 0
    score["Avg_Statistical"] = np.mean([ks_val, tv_val, corr_val, bound_val])

    # New row synthesis (% novel)
    nrs_data = res.get("NewRowSynthesis", {})
    if isinstance(nrs_data, dict):
        score["Novel_Rows_%"] = nrs_data.get("unique_rows_percentage", 0) * 100
    else:
        score["Novel_Rows_%"] = 0

    # Privacy (detection AUC, lower = harder to detect = more private)
    det = res.get("BinaryDT_Detection", 0)
    if isinstance(det, float):
        score["Detection_AUC"] = det
        score["Privacy_Score"] = 1 - det  # Higher = more private
    else:
        score["Detection_AUC"] = 0
        score["Privacy_Score"] = 0

    scorecard.append(score)

sc_df = pd.DataFrame(scorecard).set_index("Method")
print(sc_df.round(4).to_string())
print("\nKey:")
print("  Avg_Statistical: Average of KSComplement, TVComplement, CorrelationSimilarity, BoundaryAdherence (higher=better)")
print("  Novel_Rows_%: Percentage of synthetic rows that are NOT copies of real rows (higher=better)")
print("  Detection_AUC: AUC of classifier trying to distinguish real vs synthetic (lower=better)")
print("  Privacy_Score: 1 - Detection_AUC (higher=more private)")

sc_df.to_csv("F:/synthetic/results/sdmetrics_scorecard.csv")
with open("F:/synthetic/results/sdmetrics_full.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\nResults saved to /f/synthetic/results/sdmetrics_*")
print("DONE")
