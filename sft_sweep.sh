#!/bin/bash
# ============================================================
# ToolRL Quad-GPU Parallel Sweep
# Tuned for Docker setup with /workspace and /app/models
# Target: Qwen2.5-1.5B-Instruct hitting 53.60% API-Bank
# ============================================================

set -e

# ---- EDIT THESE TWO LINES ONLY ----
BASE_MODEL="/app/models/Qwen2.5-1.5B-Instruct"  # Target 1.5B for the anchor
MODEL_NAME="qwen2.5-1.5b"                       # Used for folder names
# -----------------------------------

MICRO_BATCH=8          # reduce to 4 if OOM

RESULTS_DIR="/workspace/sweep_results/${MODEL_NAME}"
CHECKPOINT_DIR="/workspace/sweep_checkpoints/${MODEL_NAME}"
TRAIN_DATA="/workspace/dataset/rlla_sft/train.parquet"
VAL_DATA="/workspace/dataset/rlla_sft/val.parquet"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR"

# Sweep grid (54 total configs)
LR_VALUES=(5e-6 1e-5 5e-5)
MAX_LEN_VALUES=(2048 4096)
EPOCH_VALUES=(3 5 7)
BATCH_VALUES=(16 32 64)

RESULTS_CSV="${RESULTS_DIR}/sweep_results.csv"

# Initialize CSV with exact columns needed by analyze_sweep.py
if [ ! -f "$RESULTS_CSV" ]; then
    echo "run_id,lr,max_length,epochs,batch_size,apibank_overall,apibank_l1,apibank_l2,apibank_l3,bfcl_overall,bamboogle,dist_to_paper,status" > "$RESULTS_CSV"
fi

# Generate all 54 configs into an array
CONFIGS=()
run_id=0
for LR in "${LR_VALUES[@]}"; do
    for MAX_LEN in "${MAX_LEN_VALUES[@]}"; do
        for EPOCHS in "${EPOCH_VALUES[@]}"; do
            for BATCH in "${BATCH_VALUES[@]}"; do
                run_id=$((run_id + 1))
                CONFIGS+=("${run_id}|${LR}|${MAX_LEN}|${EPOCHS}|${BATCH}")
            done
        done
    done
done

# Worker function to process a subset of configs on a specific GPU
run_worker() {
    local gpu_id=$1
    shift
    local my_configs=("$@")

    for config in "${my_configs[@]}"; do
        IFS='|' read -r c_id LR MAX_LEN EPOCHS BATCH <<< "$config"

        RUN_NAME="sft_lr${LR}_len${MAX_LEN}_ep${EPOCHS}_bs${BATCH}"
        CKPT_PATH="${CHECKPOINT_DIR}/${RUN_NAME}"
        LOG_FILE="${RESULTS_DIR}/${RUN_NAME}.log"

        echo "[GPU $gpu_id] ==== RUN ${c_id}/54: ${RUN_NAME} ===="

        # Force this process to only see its assigned GPU
        export CUDA_VISIBLE_DEVICES=$gpu_id

        # Skip if checkpoint already exists
        if [ -d "$CKPT_PATH" ] && [ "$(ls -A $CKPT_PATH 2>/dev/null)" ]; then
            echo "  [GPU $gpu_id SKIP] checkpoint exists: $RUN_NAME"
            continue
        fi

        # Train (Single Node, Single Process per worker)
        torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            -m verl.trainer.fsdp_sft_trainer \
            data.train_files=$TRAIN_DATA \
            data.val_files=$VAL_DATA \
            data.prompt_key=prompt \
            data.response_key=response \
            data.max_length=$MAX_LEN \
            data.truncation=right \
            data.train_batch_size=$BATCH \
            data.micro_batch_size=$MICRO_BATCH \
            model.partial_pretrain=$BASE_MODEL \
            model.enable_gradient_checkpointing=True \
            trainer.default_local_dir=$CKPT_PATH \
            trainer.project_name=toolrl-sft-sweep \
            trainer.experiment_name=$RUN_NAME \
            trainer.total_epochs=$EPOCHS \
            trainer.logger=['console','mlflow'] \
            optim.lr=$LR \
            > "$LOG_FILE" 2>&1

        # API-Bank eval
        APIBANK_OUT="${RESULTS_DIR}/apibank_${RUN_NAME}.json"
        python /workspace/benchmarks/API-Bank/generate.py \
            --model_path "$CKPT_PATH" \
            --output_path "$APIBANK_OUT" >> "$LOG_FILE" 2>&1 || true

        # Extract exactly the Overall Accuracy number
        APIBANK_SCORE=$(python /workspace/benchmarks/API-Bank/evaluate.py \
            --pred_file "$APIBANK_OUT" 2>/dev/null | grep -i "Overall Accuracy" | awk '{print $NF}' | tr -d '%' || echo "")

        if [ -z "$APIBANK_SCORE" ]; then
            APIBANK_SCORE="0.00"
            STATUS="eval_failed"
        else
            STATUS="done"
        fi

        # Append to CSV with empty commas for unused columns to align with python script
        echo "${c_id},${LR},${MAX_LEN},${EPOCHS},${BATCH},${APIBANK_SCORE},,,,,,,${STATUS}" >> "$RESULTS_CSV"
        echo "  [GPU $gpu_id DONE] api_bank=${APIBANK_SCORE}%"
    done
}

# Split the 54 configs into 4 groups (14, 14, 13, 13)
G0=("${CONFIGS[@]:0:14}")
G1=("${CONFIGS[@]:14:14}")
G2=("${CONFIGS[@]:28:13}")
G3=("${CONFIGS[@]:41:13}")

echo "Starting 4 parallel workers..."
echo "Monitor progress live by running: python3 analyze_sweep.py --watch"
echo "============================================================"

# Launch 4 parallel workers in the background
run_worker 0 "${G0[@]}" &
run_worker 1 "${G1[@]}" &
run_worker 2 "${G2[@]}" &
run_worker 3 "${G3[@]}" &

# Wait for all background jobs to finish
wait

echo ""
echo "============================================================"
echo "Sweep complete. Final results saved to: $RESULTS_CSV"
echo "============================================================"
