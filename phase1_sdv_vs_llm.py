# %% [markdown]
# # Phase 1: SDV vs LLM — Synthetic Tabular Data Generation Benchmark
# ## AutoTherm Dataset · Thermal Comfort Estimation
#
# **Experiment Goals:**
# - Dimension 1: Statistical Fidelity — distribution similarity of synthetic vs real data
# - Dimension 2: ML Utility (TSTR) — performance on real test set when trained on synthetic data
#
# **Methods Compared:**
# - SDV camp: CTGAN · TVAE · GaussianCopula · PAR
# - LLM camp: GReaT · REaLTabFormer · GPT-4o Zero-Shot
#
# > ⚠️ A **T4 GPU runtime** is recommended for Colab. This notebook defaults to 20,000 rows for quick validation.
# > For full experiments, sample 50,000–100,000 rows and mount Google Drive to save intermediate results.

# %% [markdown]
# ## 1. Environment Setup

# %%
# @title Install dependencies (~3-5 min)
import sys
import subprocess

packages = [
    "sdv>=1.10.0",          # CTGAN, TVAE, GaussianCopula, PAR
    "datasets>=2.14.0",     # HuggingFace datasets
    "be-great>=0.0.7",      # GReaT: LLM-based tabular generation
    "realtabformer>=0.1.0", # REaLTabFormer
    "peft>=0.7.0",          # Required for GReaT LoRA fine-tuning
    "openai>=1.0.0",        # GPT-4o Zero-Shot (optional)
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✅ All packages installed!")

# %% [markdown]
# ## 2. Imports & Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import warnings
import time
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

# SDV
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.sequential import PARSynthesizer

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, f1_score, accuracy_score,
    confusion_matrix, mean_squared_error, r2_score
)

# Datasets
from datasets import load_dataset

# PyTorch & Transformers (for GReaT / REaLTabFormer)
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, EarlyStoppingCallback

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

print(f"🐍 Python: {sys.version}")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '❌ No GPU'}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# %% [markdown]
# ## 3. Global Configuration

# %%
@dataclass
class Config:
    """Global experiment config — edit here to adjust scale"""
    # Data
    dataset_name: str = "kopetri/AutoTherm"
    subset: str = "combined"           # combined / indoor / vehicle
    n_samples: int = 20_000            # rows to sample (10K-50K for Colab)
    test_size: float = 0.2
    random_state: int = 42

    # Generation
    synthetic_rows: int = 20_000       # number of synthetic rows to generate

    # SDV params
    ctgan_epochs: int = 300
    tvae_epochs: int = 300

    # LLM params
    great_epochs: int = 50             # GReaT fine-tuning epochs
    use_zero_shot: bool = False        # enable GPT-4o (requires API key)

    # Output
    save_dir: str = "./phase1_results"
    drive_mount: bool = False           # mount Google Drive

config = Config()

# Read API key from Colab environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

os.makedirs(config.save_dir, exist_ok=True)
print(f"📁 Results will be saved to: {config.save_dir}")

# %% [markdown]
# ## 4. Data Loading & Preprocessing
# AutoTherm `combined` subset: ~2.83M rows, 31-dimensional features + Label (-3 ~ +3)

# %%
print(f"📥 Loading AutoTherm '{config.subset}' subset (streaming)...")

# Stream the first N rows
dataset = load_dataset(
    config.dataset_name,
    config.subset,
    streaming=True,
    split="train"
)

# Efficient sampling
rows = []
for i, sample in enumerate(tqdm(dataset, total=config.n_samples, desc="Streaming rows")):
    rows.append(sample)
    if i + 1 >= config.n_samples:
        break

df = pd.DataFrame(rows)
print(f"\n✅ Loaded {len(df):,} rows × {len(df.columns)} columns")

# Drop non-predictive identifier columns that break SDV metadata
ID_COLS = ["file_name"]
for c in ID_COLS:
    if c in df.columns:
        df = df.drop(columns=[c])
        print(f"🗑️  Dropped ID column: {c}")

# %% [markdown]
# ### 4.1 Data Exploration

# %%
print("=" * 60)
print("📋 Dataset Overview")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumn types:")
print(df.dtypes.value_counts().to_string())
print(f"\nTarget distribution (Label):")
print(df["Label"].value_counts().sort_index().to_string())
print(f"\nMissing values per column:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0].to_string())
else:
    print("  ✅ No missing values")

# %% [markdown]
# ### 4.2 Feature Engineering: Separate Continuous / Categorical / Target

# %%
def identify_column_types(df: pd.DataFrame, target_col: str = "Label") -> Tuple[List[str], List[str]]:
    """Automatically identify continuous and categorical features"""
    categorical_cols = []
    continuous_cols = []

    for col in df.columns:
        if col == target_col:
            continue
        if df[col].dtype == "object" or df[col].dtype == "string":
            categorical_cols.append(col)
        elif df[col].nunique() <= 20:  # low-cardinality numeric columns → categorical
            categorical_cols.append(col)
        else:
            continuous_cols.append(col)

    return categorical_cols, continuous_cols

TARGET = "Label"
cat_cols, cont_cols = identify_column_types(df, TARGET)

print(f"🎯 Target: '{TARGET}'")
print(f"📝 Categorical ({len(cat_cols)}): {cat_cols}")
print(f"📈 Continuous ({len(cont_cols)}): {cont_cols}")

# %% [markdown]
# ### 4.3 Data Cleaning & Splitting

# %%
# Copy and clean
df_clean = df.copy()

# Handle missing values
for col in cat_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else "missing")
for col in cont_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Convert categorical features to strings (SDV requirement)
for col in cat_cols:
    df_clean[col] = df_clean[col].astype(str)

# Label encoding (for some evaluations)
le = LabelEncoder()
df_clean["LabelEncoded"] = le.fit_transform(df_clean[TARGET].astype(str))
n_classes = len(le.classes_)

print(f"🎯 {n_classes} classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split
train_df, test_df = train_test_split(
    df_clean, test_size=config.test_size,
    random_state=config.random_state, stratify=df_clean[TARGET].astype(str)
)

X_train = train_df[cat_cols + cont_cols]
y_train = train_df[TARGET].astype(str)
X_test = test_df[cat_cols + cont_cols]
y_test = test_df[TARGET].astype(str)

print(f"\n📊 Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
print(f"📊 Train label dist:\n{train_df[TARGET].value_counts(normalize=True).sort_index()}")

# %% [markdown]
# ## 5. Data Encoding (shared across all experiments)

# %%
from sklearn.preprocessing import OrdinalEncoder

X_train_enc = X_train.copy()
X_test_enc = X_test.copy()

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_enc[cat_cols] = oe.fit_transform(X_train[cat_cols])
X_test_enc[cat_cols] = oe.transform(X_test[cat_cols])

# Target encoding
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test)

# Baseline will be measured using paper's LSTM (see indoor_frames_lstm.py / export_and_train.py)
# For this notebook, we skip the baseline and focus on synthetic data generation & statistical fidelity.

# %% [markdown]
# ## 6. Synthetic Data Generation
# ### 6.1 SDV Methods

# %%
# Prepare data for SDV (must include target column)
sdv_train = train_df[cat_cols + cont_cols + [TARGET]].copy()
for col in cat_cols:
    sdv_train[col] = sdv_train[col].astype(str)
sdv_train[TARGET] = sdv_train[TARGET].astype(str)

# Create SDV metadata
def create_metadata(df: pd.DataFrame, cat_cols: List[str], cont_cols: List[str], target: str) -> SingleTableMetadata:
    """Create SDV metadata with explicit column types"""
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)

    # Ensure target column is categorical (classification task)
    metadata.update_column(
        column_name=target,
        sdtype="categorical"
    )

    # Explicitly set column types
    for col in cat_cols:
        if col != target:
            metadata.update_column(column_name=col, sdtype="categorical")
    for col in cont_cols:
        metadata.update_column(column_name=col, sdtype="numerical")

    # Remove any auto-detected primary key (e.g. file_name)
    try:
        metadata.remove_primary_key()
    except Exception:
        pass

    return metadata

metadata = create_metadata(sdv_train, cat_cols, cont_cols, TARGET)
print("✅ SDV Metadata created:")
print(metadata)

# %% [markdown]
# ### 6.1a GaussianCopula (statistical baseline, ~1 min)

# %%
print("🟢 Training GaussianCopulaSynthesizer...")
t0 = time.time()

gc_synthesizer = GaussianCopulaSynthesizer(metadata)
gc_synthesizer.fit(sdv_train)
gc_synthetic = gc_synthesizer.sample(num_rows=config.synthetic_rows)

gc_time = time.time() - t0
print(f"✅ GaussianCopula completed in {gc_time:.1f}s")
print(f"   Synthetic shape: {gc_synthetic.shape}")

# %% [markdown]
# ### 6.1b CTGAN (GAN-based, ~5-10 min on T4)

# %%
print("🟡 Training CTGANSynthesizer...")
t0 = time.time()

ctgan_synthesizer = CTGANSynthesizer(
    metadata,
    epochs=config.ctgan_epochs,
    cuda=True if DEVICE == "cuda" else False,
    verbose=True
)
ctgan_synthesizer.fit(sdv_train)
ctgan_synthetic = ctgan_synthesizer.sample(num_rows=config.synthetic_rows)

ctgan_time = time.time() - t0
print(f"✅ CTGAN completed in {ctgan_time:.1f}s")
print(f"   Synthetic shape: {ctgan_synthetic.shape}")

# %% [markdown]
# ### 6.1c TVAE (VAE-based, ~3-8 min on T4)

# %%
print("🟠 Training TVAESynthesizer...")
t0 = time.time()

tvae_synthesizer = TVAESynthesizer(
    metadata,
    epochs=config.tvae_epochs,
    cuda=True if DEVICE == "cuda" else False,
    verbose=True
)
tvae_synthesizer.fit(sdv_train)
tvae_synthetic = tvae_synthesizer.sample(num_rows=config.synthetic_rows)

tvae_time = time.time() - t0
print(f"✅ TVAE completed in {tvae_time:.1f}s")
print(f"   Synthetic shape: {tvae_synthetic.shape}")

# %% [markdown]
# ### 6.1d PAR (Temporal Synthesis — optional, enable only when temporal data is needed)

# %%
# PAR requires a time/sequence index.
# If no native sequence_id exists, we simulate a simple scenario:
# each sample is a length-1 sequence.
# For full temporal experiments, use indoor_frames / vehicle_frames subsets.

def create_par_data(df, time_col=None):
    """Prepare data for PAR — simplified version"""
    df_par = df.copy()
    # Create dummy time column if no native one exists
    if time_col is None or time_col not in df_par.columns:
        df_par["_time_idx"] = np.arange(len(df_par))
        time_col = "_time_idx"
    # PAR requires a sequence_index column
    df_par["_sequence_idx"] = 0  # Simplified: all rows belong to a single sequence
    return df_par, time_col

print("🟣 PAR Temporal Synthesis — simplified (single sequence)")
print("⚠️  For full temporal experiments, use indoor_frames subset with native time index")
print()

try:
    par_train, par_time_col = create_par_data(sdv_train)

    # PAR requires SequentialMetadata
    from sdv.metadata import SingleTableMetadata
    par_metadata = SingleTableMetadata()
    par_metadata.detect_from_dataframe(par_train)
    par_metadata.update_column(column_name=TARGET, sdtype="categorical")
    for col in cat_cols:
        par_metadata.update_column(column_name=col, sdtype="categorical")
    for col in cont_cols:
        par_metadata.update_column(column_name=col, sdtype="numerical")
    par_metadata.set_sequence_key(column_name="_sequence_idx")
    par_metadata.set_sequence_index(column_name="_time_idx")

    t0 = time.time()
    par_synthesizer = PARSynthesizer(
        par_metadata,
        epochs=100,
        cuda=True if DEVICE == "cuda" else False,
        verbose=True
    )
    par_synthesizer.fit(par_train)
    par_synthetic = par_synthesizer.sample(num_sequences=config.synthetic_rows)
    # Remove auxiliary columns
    par_synthetic = par_synthetic.drop(columns=["_sequence_idx", "_time_idx"], errors="ignore")

    par_time = time.time() - t0
    print(f"✅ PAR completed in {par_time:.1f}s")
    print(f"   Synthetic shape: {par_synthetic.shape}")
except Exception as e:
    print(f"⚠️  PAR failed (this is OK for Phase 1): {e}")
    par_synthetic = None
    par_time = float("nan")

# %% [markdown]
# ## 6.2 LLM Methods

# %% [markdown]
# ### 6.2a GReaT — Fine-tune GPT-2 for tabular data generation

# %%
print("🔵 GReaT: Fine-tuning GPT-2 on serialized tabular data")
print(f"   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

from be_great import GReaT

# GReaT works directly on the original DataFrame (no one-hot encoding needed)
great_train = train_df[cat_cols + cont_cols + [TARGET]].copy()
for col in cat_cols:
    great_train[col] = great_train[col].astype(str)
great_train[TARGET] = great_train[TARGET].astype(str)

t0 = time.time()

try:
    # Use distilgpt2 to save GPU memory (Colab T4 15GB is sufficient)
    model = GReaT(
        llm="distilgpt2",
        epochs=config.great_epochs,
        batch_size=32,
        efficient_finetuning="lora",  # LoRA saves VRAM
        experiment_dir=os.path.join(config.save_dir, "great_checkpoints")
    )

    model.fit(great_train)
    great_synthetic = model.sample(n_samples=config.synthetic_rows)

    great_time = time.time() - t0
    print(f"✅ GReaT completed in {great_time:.1f}s")
    print(f"   Synthetic shape: {great_synthetic.shape}")
except Exception as e:
    print(f"❌ GReaT failed: {e}")
    print("   This is common in Colab due to memory constraints.")
    print("   Consider reducing n_samples or epochs.")
    great_synthetic = None
    great_time = float("nan")

# %% [markdown]
# ### 6.2b REaLTabFormer — Transformer-based tabular generation

# %%
print("🟣 REaLTabFormer: GPT-2 autoregressive table generation")

from realtabformer import REaLTabFormer

t0 = time.time()

try:
    # Prepare data in REaLTabFormer's expected format
    rtf_train = train_df[cat_cols + cont_cols + [TARGET]].copy()
    for col in cat_cols:
        rtf_train[col] = rtf_train[col].astype(str)
    rtf_train[TARGET] = rtf_train[TARGET].astype(str)

    # Ensure clean index
    rtf_train = rtf_train.reset_index(drop=True)

    rtf_model = REaLTabFormer(
        model_type="tabular",
        checkpoints_dir=os.path.join(config.save_dir, "rtf_checkpoints"),
        epochs=50,  # reduced epochs for Colab
        train_size=0.8
    )

    rtf_model.fit(rtf_train)
    rtf_synthetic = rtf_model.sample(n_samples=config.synthetic_rows, generate_kwargs={})

    rtf_time = time.time() - t0
    print(f"✅ REaLTabFormer completed in {rtf_time:.1f}s")
    print(f"   Synthetic shape: {rtf_synthetic.shape}")
except Exception as e:
    print(f"❌ REaLTabFormer failed: {e}")
    print("   This method requires significant GPU RAM.")
    rtf_synthetic = None
    rtf_time = float("nan")

# %% [markdown]
# ### 6.2c GPT-4o / LLM Zero-Shot (Optional — requires API Key)

# %%
print("🟡 LLM Zero-Shot Generation via OpenAI API")
print(f"   Enabled: {config.use_zero_shot}")

llm_synthetic = None
llm_time = float("nan")

if config.use_zero_shot and OPENAI_API_KEY:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Build prompt using real data statistics
    stats_desc = []
    for col in cont_cols[:8]:  # describe first 8 continuous columns as examples
        stats_desc.append(
            f"  - {col}: mean={train_df[col].mean():.2f}, "
            f"std={train_df[col].std():.2f}, "
            f"range=[{train_df[col].min():.2f}, {train_df[col].max():.2f}]"
        )

    cat_desc = []
    for col in cat_cols[:5]:
        unique_vals = train_df[col].unique()
        cat_desc.append(f"  - {col}: {list(unique_vals)[:10]}")

    prompt = f"""Generate {min(config.synthetic_rows, 200)} synthetic rows of tabular data
for thermal comfort prediction. Output as JSON array of objects.

Dataset context: AutoTherm thermal comfort estimation.
- Environment: indoor climatic chamber + vehicle cabin
- Target: Label (thermal comfort: -3=very cold, -2=cold, -1=slightly cool, 0=neutral, 1=slightly warm, 2=warm, 3=very hot)

Categorical features:
{chr(10).join(cat_desc)}

Continuous features (statistics from real data):
{chr(10).join(stats_desc)}

Generate realistic, diverse samples. Output ONLY the JSON array, no other text.
Each row must include all features and a plausible Label value."""

    print(f"   Prompt length: {len(prompt)} chars")
    print("   Calling GPT-4o...")

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=16000
        )
        synthetic_json = response.choices[0].message.content
        # Extract JSON from response
        import re
        json_match = re.search(r"\[.*\]", synthetic_json, re.DOTALL)
        if json_match:
            llm_synthetic = pd.DataFrame(json.loads(json_match.group()))
        else:
            llm_synthetic = pd.DataFrame(json.loads(synthetic_json))

        llm_time = time.time() - t0
        print(f"✅ LLM Zero-Shot completed in {llm_time:.1f}s")
        print(f"   Generated {len(llm_synthetic)} rows, cost ~${llm_time * 0.01:.4f}")
    except Exception as e:
        print(f"❌ LLM Zero-Shot failed: {e}")
        llm_synthetic = None
        llm_time = float("nan")
else:
    print("   ℹ️  Skipped. Set config.use_zero_shot=True and provide OPENAI_API_KEY to enable.")
    print("   In Colab: import os; os.environ['OPENAI_API_KEY'] = 'sk-...'")

# %% [markdown]
# ## 7. Collect All Synthetic Datasets

# %%
synthetic_datasets = {
    "GaussianCopula": (gc_synthetic, gc_time),
    "CTGAN": (ctgan_synthetic, ctgan_time),
    "TVAE": (tvae_synthetic, tvae_time),
    "PAR": (par_synthetic, par_time),
    "GReaT": (great_synthetic, great_time),
    "REaLTabFormer": (rtf_synthetic, rtf_time),
    "LLM-ZeroShot": (llm_synthetic, llm_time),
}

# Filter out failed methods
valid_methods = {k: v for k, v in synthetic_datasets.items() if v[0] is not None}
print(f"✅ {len(valid_methods)}/7 methods succeeded: {list(valid_methods.keys())}")

if len(valid_methods) < 2:
    raise RuntimeError("At least 2 methods must succeed to compare. Try reducing n_samples or epochs.")

# %% [markdown]
# ## 8. Dimension 1: Statistical Fidelity

# %%
from scipy import stats
from scipy.spatial.distance import jensenshannon

def calculate_statistical_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    cat_cols: List[str],
    cont_cols: List[str],
    target: str,
    n_bins: int = 50
) -> Dict[str, float]:
    """Multi-dimensional statistical fidelity evaluation"""

    metrics = {}

    # --- 1. Wasserstein Distance for continuous features ---
    w_distances = []
    for col in cont_cols + [target]:
        real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna().values
        synth_vals = pd.to_numeric(synth_df[col], errors="coerce").dropna().values
        if len(real_vals) > 0 and len(synth_vals) > 0:
            w_distances.append(stats.wasserstein_distance(real_vals, synth_vals))
    metrics["wasserstein_mean"] = np.mean(w_distances) if w_distances else float("nan")

    # --- 2. KS Statistic for continuous features ---
    ks_stats = []
    for col in cont_cols:
        real_vals = pd.to_numeric(real_df[col], errors="coerce").dropna().values
        synth_vals = pd.to_numeric(synth_df[col], errors="coerce").dropna().values
        if len(real_vals) > 0 and len(synth_vals) > 0:
            ks_stats.append(stats.ks_2samp(real_vals, synth_vals).statistic)
    metrics["ks_mean"] = np.mean(ks_stats) if ks_stats else float("nan")

    # --- 3. Jensen-Shannon Divergence for categorical features ---
    js_divs = []
    for col in cat_cols + [target]:
        real_counts = real_df[col].value_counts(normalize=True)
        synth_counts = synth_df[col].value_counts(normalize=True)
        all_cats = list(set(real_counts.index) | set(synth_counts.index))
        p = np.array([real_counts.get(c, 0) for c in all_cats])
        q = np.array([synth_counts.get(c, 0) for c in all_cats])
        js_divs.append(jensenshannon(p, q) if p.sum() > 0 and q.sum() > 0 else 0)
    metrics["js_divergence_mean"] = np.mean(js_divs) if js_divs else float("nan")

    # --- 4. Correlation matrix difference ---
    real_corr = real_df[cont_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0)
    synth_corr = synth_df[cont_cols].apply(pd.to_numeric, errors="coerce").corr().fillna(0)

    # Align columns
    common_cols = [c for c in real_corr.columns if c in synth_corr.columns]
    if len(common_cols) > 1:
        diff = (real_corr.loc[common_cols, common_cols] - synth_corr.loc[common_cols, common_cols]).abs()
        metrics["correlation_diff"] = diff.values[np.triu_indices_from(diff.values, k=1)].mean()
    else:
        metrics["correlation_diff"] = float("nan")

    # --- 5. Target distribution deviation ---
    real_target_dist = real_df[target].value_counts(normalize=True).sort_index()
    synth_target_dist = synth_df[target].value_counts(normalize=True).sort_index()
    all_labels = sorted(set(real_target_dist.index) | set(synth_target_dist.index))
    target_diff = np.mean([
        abs(real_target_dist.get(l, 0) - synth_target_dist.get(l, 0))
        for l in all_labels
    ])
    metrics["target_dist_diff"] = target_diff

    return metrics

print("📊 Calculating Statistical Fidelity for all methods...\n")

fidelity_results = {}

for method_name, (synth_df, gen_time) in tqdm(valid_methods.items(), desc="Evaluating fidelity"):
    # Align columns
    common_cols = [c for c in sdv_train.columns if c in synth_df.columns]
    real_subset = sdv_train[common_cols]
    synth_subset = synth_df[common_cols]

    # Limit comparison size
    n_compare = min(len(real_subset), len(synth_subset), 10_000)
    real_subset = real_subset.sample(n=n_compare, random_state=config.random_state)
    synth_subset = synth_subset.sample(n=n_compare, random_state=config.random_state)

    metrics = calculate_statistical_fidelity(
        real_subset, synth_subset,
        cat_cols=[c for c in cat_cols if c in common_cols],
        cont_cols=[c for c in cont_cols if c in common_cols],
        target=TARGET if TARGET in common_cols else "Label"
    )

    metrics["generation_time_s"] = gen_time
    fidelity_results[method_name] = metrics

# Convert to DataFrame
fidelity_df = pd.DataFrame(fidelity_results).T
fidelity_df.index.name = "Method"

print("\n" + "=" * 70)
print("📊 STATISTICAL FIDELITY RESULTS")
print("=" * 70)
print(fidelity_df.round(4).to_string())

# %% [markdown]
# ### 8.1 Fidelity Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics_to_plot = ["wasserstein_mean", "ks_mean", "js_divergence_mean", "correlation_diff"]
titles = [
    "Wasserstein Distance (lower = better)",
    "KS Statistic (lower = better)",
    "JS Divergence (lower = better)",
    "Correlation Diff (lower = better)"
]

for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
    if metric in fidelity_df.columns:
        vals = fidelity_df[metric].dropna().sort_values()
        colors = ["#ff6b6b" if v == vals.max() else "#51cf66" if v == vals.min() else "#4dabf7" for v in vals.values]
        ax.barh(range(len(vals)), vals.values, color=colors)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel(metric)
        # Annotate best
        ax.text(0.98, 0.02, f"Best: {vals.index[0]}", transform=ax.transAxes,
                ha="right", fontsize=9, color="green")

fig.suptitle("Statistical Fidelity: SDV vs LLM Methods", fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(config.save_dir, "fidelity_comparison.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 8.2 Target Distribution Comparison

# %%
fig, axes = plt.subplots(1, min(len(valid_methods), 4), figsize=(16, 4))

real_target_dist = sdv_train[TARGET].value_counts(normalize=True).sort_index()

for i, (method_name, (synth_df, _)) in enumerate(list(valid_methods.items())[:4]):
    ax = axes[i] if len(valid_methods) >= 4 else axes
    synth_target_dist = synth_df[TARGET].value_counts(normalize=True).sort_index() if TARGET in synth_df.columns else None

    if synth_target_dist is not None:
        all_labels = sorted(set(real_target_dist.index) | set(synth_target_dist.index))
        x = np.arange(len(all_labels))
        width = 0.35
        ax.bar(x - width/2, [real_target_dist.get(l, 0) for l in all_labels], width, label="Real", color="#4dabf7", alpha=0.8)
        ax.bar(x + width/2, [synth_target_dist.get(l, 0) for l in all_labels], width, label="Synthetic", color="#ff6b6b", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(all_labels, fontsize=8)
        ax.set_title(method_name, fontweight="bold")
        ax.legend(fontsize=8)

fig.suptitle("Target (Label) Distribution: Real vs Synthetic", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(config.save_dir, "target_distribution.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. ML Utility — TSTR (Train on Synthetic, Test on Real)
#
# TSTR evaluation uses the paper's LSTM architecture.
# See `export_and_train.py` and `indoor_frames_lstm.py` for the full 2x2 matrix pipeline.
#
# For this notebook, we report only statistical fidelity (model-free, Section 8).

# %% [markdown]
# ## 11. Key Findings Summary & Next Steps

# %%
print("""
╔══════════════════════════════════════════════════════════════╗
║           PHASE 1: KEY FINDINGS & NEXT STEPS                 ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  DIMENSION 1 — Statistical Fidelity                          ║
║    - Distribution similarity ranking (Wasserstein / KS / JS) ║
║    - Correlation matrix preservation                         ║
║    - Target distribution deviation                           ║
║                                                              ║
║  DIMENSION 2 — ML Utility                                    ║
║    - TSTR Macro-F1 comparison                                ║
║    - Per-class F1 analysis (which classes are hardest?)      ║
║    - Gap vs real-data training                               ║
║                                                              ║
║  EFFICIENCY                                                  ║
║    - Training time ranking                                   ║
║    - Cost-effectiveness analysis                             ║
║                                                              ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  → Phase 2: Privacy Evaluation (MIA, DCR, Membership Attack) ║
║  → Phase 3: Temporal Coherence (AutoTherm temporal signal)   ║
║  → Phase 4: Domain Knowledge Injection & Cross-Domain Transfer║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
