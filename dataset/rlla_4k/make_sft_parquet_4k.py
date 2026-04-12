"""
Convert rlla_rl.json (4000 samples) into parquet format for veRL fsdp_sft_trainer.
Same field structure as rlla_sft.json: instruction + input -> prompt, output -> response.
"""
import json
import pandas as pd
import os

SFT_JSON = "./dataset/rlla_4k_raw/rlla_rl.json"
OUT_DIR = "./dataset/rlla_4k"

with open(SFT_JSON) as f:
    data = json.load(f)

print(f"Total raw samples: {len(data)}")

records = []
for d in data:
    prompt = d["instruction"] + "\n\n" + d["input"]
    response = d["output"]
    records.append({"prompt": prompt, "response": response})

df = pd.DataFrame(records)
os.makedirs(OUT_DIR, exist_ok=True)

# Keep last 20 as val, rest as train
train_df = df.iloc[:-20]
val_df = df.iloc[-20:]

train_df.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUT_DIR, "val.parquet"), index=False)

print(f"Train: {len(train_df)} samples -> {OUT_DIR}/train.parquet")
print(f"Val:   {len(val_df)} samples -> {OUT_DIR}/val.parquet")
print(f"\nSample prompt (first 300 chars):\n{train_df.iloc[0]['prompt'][:300]}")
print(f"\nSample response (first 200 chars):\n{train_df.iloc[0]['response'][:200]}")
