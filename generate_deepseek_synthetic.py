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
N_GEN = 5000

# Load real data for statistics
print("\nLoading real data for prompt stats...")
ds = load_from_disk("F:/synthetic/dataset/indoor/train")
rows = [ds[i] for i in range(min(5000, len(ds)))]
real = pd.DataFrame([{f: r[f] for f in FEATURES + [TARGET]} for r in rows])
real[TARGET] = real[TARGET].astype(int)

# Build prompt
stats = []
for f in FEATURES:
    vals = pd.to_numeric(real[f], errors="coerce").dropna()
    stats.append(
        f"  {f}: min={vals.min():.2f}, max={vals.max():.2f}, "
        f"mean={vals.mean():.2f}, std={vals.std():.2f}"
    )
label_dist = real[TARGET].value_counts().sort_index().to_dict()

prompt = f"""Generate 50 synthetic data rows for thermal comfort prediction as JSON array.

Features (5 sensors):
{chr(10).join(stats)}

Label (thermal sensation): -3=very cold, -2=cold, -1=slightly cool, 0=neutral, 1=slightly warm, 2=warm, 3=very hot
Real label distribution: {label_dist}

Rules:
- High ambient_temp + high radiation → positive label
- Low wrist_skin_temp + low ambient_temp → negative label
- Feature values must be within the ranges above
- Label distribution should roughly match the real distribution

Output ONLY a JSON array. Each object: {{"Radiation-Temp": float, "Wrist_Skin_Temperature": float, "GSR": float, "Ambient_Temperature": float, "Ambient_Humidity": float, "Label": int}}"""

print(f"Prompt: {len(prompt)} chars")
print(f"Generating {N_GEN} rows in batches of 50 via DeepSeek API...")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
all_rows = []

for batch in range(N_GEN // 50):
    t0 = time.time()
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=4000
        )
        text = response.choices[0].message.content
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            rows = json.loads(match.group())
            all_rows.extend(rows)
            print(f"  Batch {batch+1}: {len(rows)} rows, {time.time()-t0:.0f}s, total={len(all_rows)}")
        else:
            print(f"  Batch {batch+1}: no JSON found, text={text[:100]}...")
    except Exception as e:
        print(f"  Batch {batch+1}: ERROR - {e}")
        time.sleep(5)

    time.sleep(0.5)  # rate limit

# Save
os.makedirs("F:/synthetic/llm_synthetic", exist_ok=True)
df = pd.DataFrame(all_rows)
df.to_csv("F:/synthetic/llm_synthetic/deepseek_train.csv", index=False)
print(f"\nSaved: {len(df)} rows to llm_synthetic/deepseek_train.csv")
if TARGET in df.columns:
    print(f"Label dist: {dict(sorted(df[TARGET].astype(int).value_counts().sort_index().items()))}")
print("DONE")
