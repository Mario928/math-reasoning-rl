#!/bin/bash
# Debug experiments requested by professor
# Experiment 1: Test if save/load corrupts the SFT model (using correct VERL format)
# Experiment 2: Proper SFT (VERL's format, 1 epoch, lr=1e-4) then PPO
set -x

echo "========================================="
echo "EXPERIMENT 1: SFT save/load sanity check"
echo "========================================="

python3 << 'PYEOF'
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'

# ---- Helper: evaluate model on 50 GSM8K test questions using chat template ----
def evaluate_model(model, tokenizer, n=50):
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    correct = 0
    total = min(n, len(dataset))
    model.eval()
    model.cuda()

    for i in range(total):
        question = dataset[i]["question"]
        gt = dataset[i]["answer"].split("####")[-1].strip().replace(",", "")

        # Use chat template (same format as VERL evaluation)
        messages = [{"role": "user", "content": question + " " + INSTRUCTION}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Extract answer after ####
        match = re.findall(r"#### (\-?[0-9\.\,]+)", response[-300:])
        if match:
            pred = match[-1].replace(",", "").replace("$", "")
            if pred == gt:
                correct += 1

    acc = correct / total * 100
    print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    return acc

# ---- Step 1: Train SFT using correct VERL format (3 epochs, lr=2e-5, full fine-tune) ----
print("\n=== Training SFT (same epochs/lr as original, but correct VERL format) ===")
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("openai/gsm8k", "main", split="train")

# Format using chat template (matching VERL's gsm8k_multiturn_sft.py)
def format_prompt(example):
    question = example['question'] + " " + INSTRUCTION
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": example['answer']}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

training_args = TrainingArguments(
    output_dir="/tmp/sft_debug",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    logging_steps=10,
    save_strategy="no",
    bf16=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=format_prompt,
)

trainer.train()

# ---- Step 2: Test BEFORE saving (model still in memory) ----
print("\n=== Testing SFT model BEFORE save (in memory) ===")
acc_before = evaluate_model(model, tokenizer, n=50)

# ---- Step 3: Save to disk ----
print("\n=== Saving model to disk ===")
model.save_pretrained("/tmp/sft_debug_saved")
tokenizer.save_pretrained("/tmp/sft_debug_saved")

# ---- Step 4: Load back from disk ----
print("\n=== Loading model back from disk ===")
model_loaded = AutoModelForCausalLM.from_pretrained("/tmp/sft_debug_saved", torch_dtype=torch.bfloat16)
tokenizer_loaded = AutoTokenizer.from_pretrained("/tmp/sft_debug_saved")

# ---- Step 5: Test AFTER loading ----
print("\n=== Testing SFT model AFTER load (from disk) ===")
acc_after = evaluate_model(model_loaded, tokenizer_loaded, n=50)

# ---- Step 6: Compare ----
print("\n=== EXPERIMENT 1 RESULTS ===")
print(f"Accuracy BEFORE save: {acc_before:.1f}%")
print(f"Accuracy AFTER load:  {acc_after:.1f}%")
if abs(acc_before - acc_after) < 0.1:
    print("CONCLUSION: Save/load does NOT corrupt the model. They are identical.")
else:
    print(f"CONCLUSION: Save/load CHANGES the model! Difference: {abs(acc_before - acc_after):.1f}%")

# Cleanup
del model, model_loaded, trainer
torch.cuda.empty_cache()
PYEOF

echo ""
echo "========================================="
echo "EXPERIMENT 2: Proper SFT (VERL format, 1 epoch, lr=1e-4) then PPO"
echo "========================================="

# ---- Step 1: Preprocess SFT data using VERL's format ----
echo "=== Preprocessing SFT data (VERL chat template format) ==="
python3 /tmp/verl/examples/data_preprocess/gsm8k_multiturn_sft.py \
    --local_save_dir /app/data/gsm8k_sft

# ---- Step 2: Run SFT using VERL's own trainer (1 epoch, lr=1e-4, matching their script) ----
echo ""
echo "=== Running VERL SFT (1 epoch, lr=1e-4) ==="
torchrun --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.sft_trainer \
    data.train_files=/app/data/gsm8k_sft/train.parquet \
    data.val_files=/app/data/gsm8k_sft/test.parquet \
    data.messages_key=messages \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=1e-4 \
    engine=fsdp \
    model.path=Qwen/Qwen2.5-0.5B-Instruct \
    trainer.default_local_dir=/app/models/sft_qwen_verl \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-verl-format \
    trainer.logger=console \
    trainer.total_epochs=1

# ---- Step 3: Find the saved SFT model path ----
# VERL saves to default_local_dir/global_step_XXX/huggingface/
SFT_MODEL_PATH=$(find /app/models/sft_qwen_verl -name "huggingface" -type d | sort | tail -1)
echo "SFT model saved at: $SFT_MODEL_PATH"

if [ -z "$SFT_MODEL_PATH" ]; then
    echo "ERROR: Could not find SFT model checkpoint!"
    exit 1
fi

# ---- Step 4: Run PPO on VERL-format SFT model ----
echo ""
echo "=== Running PPO on VERL-format SFT model ==="

export MLFLOW_TRACKING_URI=http://mlflow:5000

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=/app/data/gsm8k/train.parquet \
    data.val_files=/app/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=$SFT_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=$SFT_MODEL_PATH \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=True \
    trainer.logger='["console","mlflow"]' \
    trainer.project_name=verl_gsm8k \
    trainer.experiment_name=ppo_sft_verl_format \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=15

echo ""
echo "========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================="
