"""
Convert rlla_sft.json (400 samples with DeepSeek-R1 distilled <think> fields)
into parquet format for the veRL fsdp_sft_trainer.
"""
import json
import pandas as pd
import os

SFT_JSON = "./dataset/rlla_4k_raw/rlla_sft.json"
OUT_DIR = "./dataset/rlla_sft"

with open(SFT_JSON) as f:
    data = json.load(f)

print(f"Total raw SFT samples: {len(data)}")

records = []
for d in data:
    # instruction = system prompt (tool list + format instructions)
    # input = user turn (dialogue history + current query)
    # output = full model response (includes <think>...</think> from DeepSeek-R1)
    prompt = d["instruction"] + "\n\n" + d["input"]
    response = d["output"]
    records.append({"prompt": prompt, "response": response})

df = pd.DataFrame(records)
os.makedirs(OUT_DIR, exist_ok=True)

# SFT400: all 400 samples
train_df = df.iloc[:-20]
val_df = df.iloc[-20:]

train_df.to_parquet(os.path.join(OUT_DIR, "train.parquet"), index=False)
val_df.to_parquet(os.path.join(OUT_DIR, "val.parquet"), index=False)

print(f"Train: {len(train_df)} samples -> {OUT_DIR}/train.parquet")
print(f"Val:   {len(val_df)} samples -> {OUT_DIR}/val.parquet")
print(f"\nSample prompt (first 300 chars):\n{train_df.iloc[0]['prompt'][:300]}")
print(f"\nSample response (first 200 chars):\n{train_df.iloc[0]['response'][:200]}")
