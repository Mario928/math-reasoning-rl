#!/bin/bash
# SFT hyperparameter sweep: train + evaluate 6 configs
# Config 1 already trained at /app/models/sft_qwen_verl — just eval it
# Configs 2-6 train fresh, save, eval
set -x

INSTRUCTION='Let'\''s think step by step and output the final answer after "####".'
DATA_DIR=/app/data/gsm8k_sft
RESULTS_FILE=/tmp/sft_sweep_results.txt

# Make sure SFT data exists
if [ ! -f "$DATA_DIR/train.parquet" ]; then
    echo "Preprocessing SFT data..."
    python3 /tmp/verl/examples/data_preprocess/gsm8k_multiturn_sft.py \
        --local_save_dir $DATA_DIR
fi

# Evaluation function — runs on full GSM8K test set (1319 questions)
eval_model() {
    local MODEL_PATH=$1
    local CONFIG_NAME=$2
    echo ""
    echo "=== Evaluating: $CONFIG_NAME ==="
    echo "Model path: $MODEL_PATH"

    PYTHONUNBUFFERED=1 python3 << PYEOF
import torch, re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

INSTRUCTION = 'Let\\'s think step by step and output the final answer after "####".'
model = AutoModelForCausalLM.from_pretrained("$MODEL_PATH", torch_dtype=torch.bfloat16).cuda()
tokenizer = AutoTokenizer.from_pretrained("$MODEL_PATH")
model.eval()
dataset = load_dataset("openai/gsm8k", "main", split="test")
correct = 0
total = len(dataset)
for i in range(total):
    question = dataset[i]["question"]
    gt = dataset[i]["answer"].split("####")[-1].strip().replace(",", "")
    messages = [{"role": "user", "content": question + " " + INSTRUCTION}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    match = re.findall(r"#### (\-?[0-9\.\,]+)", response[-300:])
    if match:
        pred = match[-1].replace(",", "").replace("\$", "")
        if pred == gt:
            correct += 1
    if (i+1) % 200 == 0:
        print(f"  Progress: {i+1}/{total}, acc so far: {correct}/{i+1} = {correct/(i+1)*100:.1f}%")
acc = correct/total*100
print(f"RESULT $CONFIG_NAME: {correct}/{total} = {acc:.1f}%")
with open("$RESULTS_FILE", "a") as f:
    f.write(f"$CONFIG_NAME: {correct}/{total} = {acc:.1f}%\n")
PYEOF

    # cleanup GPU memory
    python3 -c "import torch; torch.cuda.empty_cache()"
}

# Train SFT helper
train_sft() {
    local SAVE_DIR=$1
    local EPOCHS=$2
    local LR=$3
    local CONFIG_NAME=$4
    local EXTRA_ARGS=$5

    echo ""
    echo "=== Training SFT: $CONFIG_NAME (epochs=$EPOCHS, lr=$LR) ==="

    # Clean previous run if exists
    rm -rf $SAVE_DIR

    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
        -m verl.trainer.sft_trainer \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.messages_key=messages \
        data.micro_batch_size_per_gpu=4 \
        optim.lr=$LR \
        engine=fsdp \
        model.path=Qwen/Qwen2.5-0.5B-Instruct \
        trainer.default_local_dir=$SAVE_DIR \
        trainer.project_name=gsm8k-sft-sweep \
        trainer.experiment_name=$CONFIG_NAME \
        trainer.logger=console \
        trainer.total_epochs=$EPOCHS \
        'checkpoint.save_contents=["model","optimizer","extra","hf_model"]' \
        $EXTRA_ARGS
}

# Find HF model path from VERL checkpoint dir
find_hf_model() {
    find $1 -name "huggingface" -type d | sort | tail -1
}

# Clear results file
echo "=== SFT Sweep Results ===" > $RESULTS_FILE
echo "Started: $(date)" >> $RESULTS_FILE
echo "" >> $RESULTS_FILE

########################################
# Config 1: 1 epoch, lr=1e-4, full FT (already trained)
########################################
echo "========================================="
echo "CONFIG 1: 1 epoch, lr=1e-4, Full FT (already trained)"
echo "========================================="
CFG1_PATH=$(find_hf_model /app/models/sft_qwen_verl)
if [ -z "$CFG1_PATH" ]; then
    echo "Config 1 model not found, training..."
    train_sft /app/models/sft_qwen_verl 1 1e-4 "cfg1_ep1_lr1e-4_full"
    CFG1_PATH=$(find_hf_model /app/models/sft_qwen_verl)
fi
eval_model "$CFG1_PATH" "cfg1_ep1_lr1e-4_full"

########################################
# Config 2: 1 epoch, lr=1e-5, full FT
########################################
echo "========================================="
echo "CONFIG 2: 1 epoch, lr=1e-5, Full FT"
echo "========================================="
train_sft /app/models/sft_sweep_cfg2 1 1e-5 "cfg2_ep1_lr1e-5_full"
CFG2_PATH=$(find_hf_model /app/models/sft_sweep_cfg2)
eval_model "$CFG2_PATH" "cfg2_ep1_lr1e-5_full"

########################################
# Config 3: 3 epochs, lr=1e-4, full FT
########################################
echo "========================================="
echo "CONFIG 3: 3 epochs, lr=1e-4, Full FT"
echo "========================================="
train_sft /app/models/sft_sweep_cfg3 3 1e-4 "cfg3_ep3_lr1e-4_full"
CFG3_PATH=$(find_hf_model /app/models/sft_sweep_cfg3)
eval_model "$CFG3_PATH" "cfg3_ep3_lr1e-4_full"

########################################
# Config 4: 3 epochs, lr=1e-5, full FT
########################################
echo "========================================="
echo "CONFIG 4: 3 epochs, lr=1e-5, Full FT"
echo "========================================="
train_sft /app/models/sft_sweep_cfg4 3 1e-5 "cfg4_ep3_lr1e-5_full"
CFG4_PATH=$(find_hf_model /app/models/sft_sweep_cfg4)
eval_model "$CFG4_PATH" "cfg4_ep3_lr1e-5_full"

########################################
# Config 5: 1 epoch, lr=1e-4, LoRA (r=32)
########################################
echo "========================================="
echo "CONFIG 5: 1 epoch, lr=1e-4, LoRA r=32"
echo "========================================="
train_sft /app/models/sft_sweep_cfg5 1 1e-4 "cfg5_ep1_lr1e-4_lora" \
    "model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear"
CFG5_PATH=$(find_hf_model /app/models/sft_sweep_cfg5)
eval_model "$CFG5_PATH" "cfg5_ep1_lr1e-4_lora"

########################################
# Config 6: 3 epochs, lr=1e-4, LoRA (r=32)
########################################
echo "========================================="
echo "CONFIG 6: 3 epochs, lr=1e-4, LoRA r=32"
echo "========================================="
train_sft /app/models/sft_sweep_cfg6 3 1e-4 "cfg6_ep3_lr1e-4_lora" \
    "model.lora_rank=32 model.lora_alpha=16 model.target_modules=all-linear"
CFG6_PATH=$(find_hf_model /app/models/sft_sweep_cfg6)
eval_model "$CFG6_PATH" "cfg6_ep3_lr1e-4_lora"

########################################
# Summary
########################################
echo ""
echo "========================================="
echo "ALL CONFIGS COMPLETE"
echo "========================================="
echo "Finished: $(date)" >> $RESULTS_FILE
cat $RESULTS_FILE
