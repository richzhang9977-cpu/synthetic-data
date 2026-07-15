"""Generate synthetic tabular data via DeepSeek API (zero-shot, no training)"""
import json, time, os, re, numpy as np, pandas as pd
from datasets import load_from_disk
from openai import OpenAI

api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key:
    print("Set DEEPSEEK_API_KEY environment variable")
    exit(1)

print("=" * 60)
print("DeepSeek API -- Synthetic Tabular Data")
print("=" * 60)

FEATURES = ["Radiation-Temp", "Wrist_Skin_Temperature", "GSR",
            "Ambient_Temperature", "Ambient_Humidity"]
TARGET = "Label"
N_GEN = 14000  # top up existing 36K to 50K
N_PER_BATCH = 200
N_SAMPLE = 300

# Load real data for statistics
print("\nLoading real data for balanced few-shot sampling...")
ds = load_from_disk("F:/synthetic/dataset/indoor/train")

# Balanced sample: 100 rows per label × 7 labels = 700 rows
from collections import defaultdict
label_bins = defaultdict(list)
ALL_LABELS = set(range(-3, 4))
for i in range(len(ds)):
    r = ds[i]
    lbl = int(r[TARGET])
    if lbl in ALL_LABELS and len(label_bins[lbl]) < 100:
        label_bins[lbl].append(r)
    if len(label_bins) == 7 and all(len(v) >= 100 for v in label_bins.values()):
        break

found_labels = sorted(label_bins.keys())
print(f"   Found {len(found_labels)}/7 labels: {found_labels}")
if len(found_labels) < 7:
    print(f"   WARNING: Only {len(found_labels)} labels in dataset. Using all available.")
balanced_rows = [r for v in label_bins.values() for r in v]
real_sample = pd.DataFrame([{f: r[f] for f in FEATURES + [TARGET]} for r in balanced_rows])
real_sample[TARGET] = real_sample[TARGET].astype(int)
print(f"   Sampled {len(real_sample)} rows, labels: {dict(sorted(real_sample[TARGET].value_counts().to_dict().items()))}")

# Per-batch sampling: different rows each time for diversity
BATCH_SIZE = N_GEN // N_PER_BATCH
print(f"   {BATCH_SIZE} batches × {N_PER_BATCH} rows/batch = {N_GEN} total rows")

prompt_template = """You are generating synthetic thermal comfort data. Below are {n_sample} REAL data rows (CSV format).
Study the patterns between features and labels. Then generate {n_gen} NEW diverse rows as a JSON array.

--- REAL DATA ---
{sample_data}
--- END REAL DATA ---

Columns:
- Radiation-Temp (C), Wrist_Skin_Temperature (C), GSR (uS), Ambient_Temperature (C), Ambient_Humidity (%)
- Label: thermal comfort (-3=cold, -2=cool, -1=slightly cool, 0=neutral, 1=slightly warm, 2=warm, 3=hot)

Key relationships: Higher ambient_temp + radiation -> warmer label. Lower wrist_temp -> colder label.
GSR rises with discomfort. Humidity amplifies thermal sensation.

Generate {n_gen} diverse rows across MULTIPLE comfort levels. NOT exact copies.
Output ONLY a JSON array. Format: {{"Radiation-Temp": float, ..., "Label": int}}"""

print(f"Generating {N_GEN} rows in {BATCH_SIZE} batches...")
print(f"  {N_SAMPLE} examples/prompt × {N_PER_BATCH} rows/batch")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
all_rows = []
rng = np.random.RandomState(42)

for batch in range(BATCH_SIZE):
    # Random sample different rows each batch for diversity
    sample = real_sample.sample(n=N_SAMPLE, replace=True, random_state=batch)
    csv_str = sample.to_csv(index=False)
    prompt = prompt_template.format(n_sample=len(sample), n_gen=N_PER_BATCH, sample_data=csv_str)

    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=16000
        )
        text = response.choices[0].message.content
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            rows = json.loads(match.group())
            all_rows.extend(rows)
            print(f"  Batch {batch+1}/{BATCH_SIZE}: {len(rows)} rows, {time.time()-t0:.0f}s, total={len(all_rows)}")
        else:
            print(f"  Batch {batch+1}: no JSON found")
    except Exception as e:
        print(f"  Batch {batch+1}: ERROR - {e}")
        time.sleep(5)
    time.sleep(0.3)

# Save (append to existing if present)
os.makedirs("F:/synthetic/llm_synthetic", exist_ok=True)
df = pd.DataFrame(all_rows)
path = "F:/synthetic/llm_synthetic/deepseek_train.csv"
if os.path.exists(path) and len(df) < N_GEN:
    existing = pd.read_csv(path)
    df = pd.concat([existing, df], ignore_index=True)
df.to_csv(path, index=False)
print(f"\nSaved: {len(df)} rows to llm_synthetic/deepseek_train.csv")
if TARGET in df.columns:
    print(f"Label dist: {dict(sorted(df[TARGET].astype(int).value_counts().sort_index().items()))}")
print("DONE")
