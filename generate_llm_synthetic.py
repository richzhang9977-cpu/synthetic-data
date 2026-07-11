"""Generate LLM-based synthetic data from indoor dataset for comparison with SDV.
Methods: GReaT (fine-tuned GPT-2), REaLTabFormer (GPT-2 autoregressive)
"""
import warnings; warnings.filterwarnings("ignore")
import time, os, re, json, numpy as np, pandas as pd
import torch
from datasets import load_from_disk
from sklearn.preprocessing import LabelEncoder

print("=" * 60)
print("LLM SYNTHETIC DATA GENERATION")
print("=" * 60)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# ============================================================
# Config
# ============================================================
FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature", "GSR",
            "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"
DATA_DIR = "F:/synthetic/dataset/indoor"
OUTPUT_DIR = "F:/synthetic/llm_synthetic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# SDV used 50K rows; LLM methods need less due to VRAM limits
N_LLM_TRAIN = 10000  # rows for LLM training (distilgpt2 + LoRA fits 6GB)
N_LLM_GEN_TRAIN = 50000  # synthetic rows to generate for training
N_LLM_GEN_TEST = 15000   # synthetic rows to generate for testing

# ============================================================
# 1. Load & Preprocess (same as simple_2x2.py)
# ============================================================
print("\n[1/4] Loading indoor dataset & preprocessing...")

ds_train = load_from_disk(f"{DATA_DIR}/train")
ds_test  = load_from_disk(f"{DATA_DIR}/test")

# Group train frames by session
train_sessions = {}
for r in ds_train:
    fn = r.get("file_name", "")
    sess = fn.split("/")[0] if "/" in fn else "unknown"
    train_sessions.setdefault(sess, []).append(r)

# Flatten training frames for tabular generation
train_frames = []
for sess, frames in train_sessions.items():
    frames.sort(key=lambda f: int(re.search(r'frame_f_(\d+)', f["file_name"]).group(1)
                                  if re.search(r'frame_f_(\d+)', f["file_name"]) else 0))
    train_frames.extend(frames)

train_flat = pd.DataFrame(train_frames)[FEATURES + [TARGET]]
train_flat = train_flat.apply(pd.to_numeric, errors="coerce").fillna(0)
train_flat[TARGET] = train_flat[TARGET].astype(int)

# Sample for LLM training (VRAM limited)
llm_train = train_flat.sample(n=min(N_LLM_TRAIN, len(train_flat)), random_state=42)
llm_train[TARGET] = llm_train[TARGET].astype(str)

le = LabelEncoder()
all_labels = sorted(set(int(f[TARGET]) for f in train_frames))
for f in train_frames:
    if int(f.get(TARGET, 0)) not in all_labels:
        all_labels.append(int(f.get(TARGET, 0)))
all_labels = sorted(all_labels)
le.fit([str(l) for l in all_labels])

print(f"   Train flat: {len(train_flat)} rows")
print(f"   LLM training sample: {len(llm_train)} rows")
print(f"   Classes: {len(le.classes_)} | Labels: {list(le.classes_)}")

# ============================================================
# 2. GReaT (fine-tuned distilgpt2)
# ============================================================
print("\n[2/4] GReaT: fine-tuning distilgpt2...")

try:
    from be_great import GReaT

    llm_train_great = llm_train.copy()
    for c in FEATURES + [TARGET]:
        llm_train_great[c] = llm_train_great[c].astype(str)

    t0 = time.time()
    great_model = GReaT(
        llm="distilgpt2",
        epochs=30,
        batch_size=8,
        efficient_finetuning="lora",
        experiment_dir=f"{OUTPUT_DIR}/great_checkpoints"
    )
    great_model.fit(llm_train_great)

    # Generate TWO independent sets
    print("   Generating synthetic train set...")
    df_great_train = great_model.sample(n_samples=N_LLM_GEN_TRAIN)
    print("   Generating synthetic test set...")
    df_great_test = great_model.sample(n_samples=N_LLM_GEN_TEST)

    # Save
    df_great_train.to_csv(f"{OUTPUT_DIR}/great_train.csv", index=False)
    df_great_test.to_csv(f"{OUTPUT_DIR}/great_test.csv", index=False)
    print(f"   GReaT done in {time.time()-t0:.1f}s")
    print(f"   Saved: great_train.csv ({df_great_train.shape}), great_test.csv ({df_great_test.shape})")
except Exception as e:
    print(f"   GReaT FAILED: {e}")
    print("   This may be due to VRAM limits. Try Colab or reduce N_LLM_TRAIN.")

# ============================================================
# 3. REaLTabFormer
# ============================================================
print("\n[3/4] REaLTabFormer: GPT-2 autoregressive...")

try:
    from realtabformer import REaLTabFormer

    rtf_train = llm_train.copy()
    for c in FEATURES + [TARGET]:
        rtf_train[c] = rtf_train[c].astype(str)
    rtf_train = rtf_train.reset_index(drop=True)

    t0 = time.time()
    rtf_model = REaLTabFormer(
        model_type="tabular",
        checkpoints_dir=f"{OUTPUT_DIR}/rtf_checkpoints",
        epochs=5,
        train_size=0.8
    )
    rtf_model.fit(rtf_train)

    print("   Generating synthetic train set...")
    df_rtf_train = rtf_model.sample(n_samples=N_LLM_GEN_TRAIN)
    print("   Generating synthetic test set...")
    df_rtf_test = rtf_model.sample(n_samples=N_LLM_GEN_TEST)

    df_rtf_train.to_csv(f"{OUTPUT_DIR}/realtabformer_train.csv", index=False)
    df_rtf_test.to_csv(f"{OUTPUT_DIR}/realtabformer_test.csv", index=False)
    print(f"   REaLTabFormer done in {time.time()-t0:.1f}s")
    print(f"   Saved: realtabformer_train.csv ({df_rtf_train.shape}), realtabformer_test.csv ({df_rtf_test.shape})")
except Exception as e:
    print(f"   REaLTabFormer FAILED: {e}")
    print("   This may be due to VRAM limits.")

# ============================================================
# 4. Summary
# ============================================================
print(f"\n[4/4] LLM synthetic data saved to {OUTPUT_DIR}/")
print("Files ready for 2x2 matrix evaluation:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    if f.endswith(".csv"):
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}") / 1024
        print(f"  {f}: {size:.0f} KB")

# Save stats for reproducibility
stats = {
    "features": FEATURES,
    "n_train_frames": len(train_flat),
    "n_llm_train_rows": len(llm_train),
    "n_synthetic_train": N_LLM_GEN_TRAIN,
    "n_synthetic_test": N_LLM_GEN_TEST,
    "n_classes": len(le.classes_),
    "labels": list(le.classes_),
}
with open(f"{OUTPUT_DIR}/generation_config.json", "w") as f:
    json.dump(stats, f, indent=2)
print("\nDONE")
