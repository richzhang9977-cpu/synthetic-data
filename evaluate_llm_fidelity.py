"""Evaluate LLM synthetic data fidelity using paper's preprocessing."""
import warnings; warnings.filterwarnings("ignore")
import os, json, time, numpy as np, pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from datasets import load_from_disk

print("=" * 60)
print("LLM SYNTHETIC DATA — Statistical Fidelity")
print("=" * 60)

# Config
FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature", "GSR",
            "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"
DATA_DIR = "F:/synthetic/dataset/indoor"
LLM_DIR = "F:/synthetic/llm_synthetic"
SDV_DIR = "F:/synthetic"  # for CTGAN/TVAE/GaussianCopula comparison

# Paper normalization bounds
PAPER_BOUNDS = {
    "Radiation-Temp": (15.0, 35.0),
    "Wrist_Skin_Temperature": (None, None),
    "GSR": (None, None),
    "Ambient_Temperature": (15.0, 40.0),
    "Ambient_Humidity": (0.0, 100.0),
}

def paper_normalize(df, bounds):
    df_n = df.copy()
    for feat, (lo, hi) in bounds.items():
        vals = pd.to_numeric(df_n[feat], errors="coerce").fillna(0)
        if lo is None or hi is None: lo, hi = vals.min(), vals.max()
        df_n[feat] = np.clip((vals - lo) / (hi - lo + 1e-8), 0, 1)
    return df_n

# Load real data
print("\n[1/3] Loading real data...")
ds_train = load_from_disk(f"{DATA_DIR}/train")
real = pd.DataFrame([{f: r[f] for f in FEATURES + [TARGET]} for r in ds_train.select(range(min(50000, len(ds_train))))])
real[TARGET] = real[TARGET].astype(int).astype(str)
real = real.apply(pd.to_numeric, errors="coerce").fillna(0)
real_n = paper_normalize(real, PAPER_BOUNDS)
print(f"   Real: {len(real)} rows")

# Load SDV synthetic data for comparison
print("\n[2/3] Loading synthetic data...")
synth_sources = {}

# SDV data
for name in ["GaussianCopula", "CTGAN", "TVAE"]:
    path = f"{SDV_DIR}/synth_indoor_paper_{name}.csv"
    if os.path.exists(path):
        synth_sources[f"SDV-{name}"] = pd.read_csv(path)
        print(f"   SDV-{name}: {synth_sources[f'SDV-{name}'].shape}")

# LLM data
for name, filename in [("GReaT", "great"), ("REaLTabFormer", "realtabformer"), ("DeepSeek", "deepseek")]:
    path = f"{LLM_DIR}/{filename}_train.csv"
    if os.path.exists(path):
        synth_sources[f"LLM-{name}"] = pd.read_csv(path)
        print(f"   LLM-{name}: {synth_sources[f'LLM-{name}'].shape}")

# ============================================================
# 3. Compute fidelity metrics
# ============================================================
print("\n[3/3] Computing fidelity metrics...")

def compute_fidelity(real, synth):
    """Same metrics as SDMetrics: Wasserstein, KS, JS, Correlation"""
    r = {}
    # Wasserstein (continuous only)
    wd = []
    for feat in FEATURES:
        rv = pd.to_numeric(real[feat], errors="coerce").dropna().values
        sv = pd.to_numeric(synth[feat], errors="coerce").dropna().values
        if len(rv) > 0 and len(sv) > 0:
            wd.append(stats.wasserstein_distance(rv, sv))
    r["wasserstein_mean"] = np.mean(wd) if wd else np.nan

    # KS
    ks_list = []
    for feat in FEATURES:
        rv = pd.to_numeric(real[feat], errors="coerce").dropna().values
        sv = pd.to_numeric(synth[feat], errors="coerce").dropna().values
        if len(rv) > 0 and len(sv) > 0:
            ks_list.append(stats.ks_2samp(rv, sv).statistic)
    r["ks_mean"] = np.mean(ks_list) if ks_list else np.nan

    # Wasserstein per feature
    for feat in FEATURES:
        rv = pd.to_numeric(real[feat], errors="coerce").dropna()
        sv = pd.to_numeric(synth[feat], errors="coerce").dropna()
        if len(rv) > 0 and len(sv) > 0:
            r[f"{feat}_w"] = stats.wasserstein_distance(rv, sv)

    # JS Divergence (target)
    rc = real[TARGET].value_counts(normalize=True)
    sc = synth[TARGET].astype(str).value_counts(normalize=True)
    all_c = list(set(rc.index) | set(sc.index))
    p = np.array([rc.get(c, 0) for c in all_c])
    q = np.array([sc.get(c, 0) for c in all_c])
    r["js_target"] = jensenshannon(p, q)

    # Correlation diff
    rcorr = real[FEATURES].corr().fillna(0)
    scorr = synth[FEATURES].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
    diff = (rcorr - scorr).abs()
    r["corr_diff"] = diff.values[np.triu_indices_from(diff.values, k=1)].mean()

    return r

# Evaluate all
n_cmp = min(len(real), 10000)
real_sample = real_n.sample(n=n_cmp, random_state=42)

results = {}
for name, syn_df in synth_sources.items():
    syn_clean = syn_df.copy()
    syn_clean[TARGET] = syn_clean[TARGET].astype(str)
    syn_clean = syn_clean.apply(pd.to_numeric, errors="coerce").fillna(0)
    syn_n = paper_normalize(syn_clean, PAPER_BOUNDS)
    syn_sample = syn_n.sample(n=min(len(syn_n), n_cmp), random_state=42)

    results[name] = compute_fidelity(real_sample, syn_sample)
    print(f"   {name:25s} | W={results[name]['wasserstein_mean']:.4f} | KS={results[name]['ks_mean']:.4f} | JS={results[name]['js_target']:.4f} | CorrDiff={results[name]['corr_diff']:.4f}")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 70)
print("STATISTICAL FIDELITY — SDV vs LLM Methods")
print("=" * 70)
cols = ["wasserstein_mean", "ks_mean", "js_target", "corr_diff"]
print(f"{'Method':>25s} | {'W_dist':>8s} | {'KS':>8s} | {'JS':>8s} | {'CorrDiff':>8s}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:>25s} | {r['wasserstein_mean']:8.4f} | {r['ks_mean']:8.4f} | {r['js_target']:8.4f} | {r['corr_diff']:8.4f}")

# Rankings
print("\nRankings (1=best):")
for metric, label in [("wasserstein_mean", "Wasserstein"), ("ks_mean", "KS"), ("js_target", "JS"), ("corr_diff", "CorrDiff")]:
    ranked = sorted(results.items(), key=lambda x: x[1][metric])
    print(f"  {label}: {' > '.join([f'{n}({r[metric]:.3f})' for n, r in ranked])}")

# Per-feature breakdown
print(f"\n{'Feat Wasserstein':>25s}", end="")
for name in results: print(f" | {name:>22s}", end="")
print()
for feat in FEATURES:
    print(f"{feat:>25s}", end="")
    for name in results:
        val = results[name].get(f"{feat}_w", 0)
        print(f" | {val:22.4f}", end="")
    print()

# Save
os.makedirs("F:/synthetic/results", exist_ok=True)
pd.DataFrame(results).T.to_csv("F:/synthetic/results/llm_fidelity.csv")
print("\nSaved: results/llm_fidelity.csv")
print("DONE")
