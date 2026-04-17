#!/bin/bash
# 3B SFT Sweep — with intermediate checkpoint cleanup + generate_batch eval

BASE_MODEL="/app/models/Qwen2.5-3B-Instruct"
MODEL_NAME="qwen2.5-3b"
MICRO_BATCH=4

RESULTS_DIR="/workspace/sweep_results/${MODEL_NAME}"
CHECKPOINT_DIR="/workspace/sweep_checkpoints/${MODEL_NAME}"
TRAIN_DATA="/workspace/dataset/rlla_sft/train.parquet"
VAL_DATA="/workspace/dataset/rlla_sft/val.parquet"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR"

LR_VALUES=(5e-6 1e-5 5e-5)
MAX_LEN_VALUES=(2048 4096)
EPOCH_VALUES=(3 5 7)
BATCH_VALUES=(16 32 64)

RESULTS_CSV="${RESULTS_DIR}/sweep_results.csv"
if [ ! -f "$RESULTS_CSV" ]; then
    echo "run_name,lr,max_length,epochs,batch_size,overall_acc,lv1_acc,lv2_acc,lv3_acc,correct_lv1,correct_lv2,correct_lv3,total_lv1,total_lv2,total_lv3,val_loss" > "$RESULTS_CSV"
fi

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

delete_intermediate_checkpoints() {
    local ckpt_path="$1"
    local latest
    latest=$(ls "$ckpt_path" | grep global_step | sort -t_ -k3 -n | tail -1)
    for step_dir in "$ckpt_path"/global_step_*/; do
        local step
        step=$(basename "$step_dir")
        if [ "$step" != "$latest" ]; then
            rm -rf "$step_dir"
        fi
    done
}

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

        export CUDA_VISIBLE_DEVICES=$gpu_id

        # Skip if checkpoint already exists
        if [ -d "$CKPT_PATH" ] && [ "$(ls -A "$CKPT_PATH" 2>/dev/null)" ]; then
            echo "  [GPU $gpu_id SKIP] checkpoint exists: $RUN_NAME"
            continue
        fi

        # Reduce micro batch for high-LR + long-seq configs to avoid OOM on 3B
        EFFECTIVE_MICRO_BATCH=$MICRO_BATCH
        if [ "$MAX_LEN" = "4096" ] && [ "$LR" = "5e-5" ]; then
            EFFECTIVE_MICRO_BATCH=2
        fi

        torchrun --standalone --nnodes=1 --nproc_per_node=1 \
            -m verl.trainer.fsdp_sft_trainer \
            data.train_files=$TRAIN_DATA \
            data.val_files=$VAL_DATA \
            data.prompt_key=prompt \
            data.response_key=response \
            data.max_length=$MAX_LEN \
            data.truncation=right \
            data.train_batch_size=$BATCH \
            data.micro_batch_size=$EFFECTIVE_MICRO_BATCH \
            model.partial_pretrain=$BASE_MODEL \
            model.enable_gradient_checkpointing=True \
            trainer.default_local_dir=$CKPT_PATH \
            trainer.project_name=toolrl-sft-sweep-3b \
            trainer.experiment_name=$RUN_NAME \
            trainer.total_epochs=$EPOCHS \
            trainer.logger=['console'] \
            optim.lr=$LR \
            > "$LOG_FILE" 2>&1 || true

        # Delete intermediate checkpoints immediately to save disk
        delete_intermediate_checkpoints "$CKPT_PATH"

        # API-Bank eval using generate_batch (fast, single GPU)
        SCORE_ROOT="/workspace/sweep_results/${MODEL_NAME}/apibank_scores"
        mkdir -p "$SCORE_ROOT"
        FINAL_STEP=$(ls "$CKPT_PATH" | grep global_step | sort -t_ -k3 -n | tail -1)
        FINAL_CKPT="$CKPT_PATH/$FINAL_STEP"

        PATH_TO_YOUR_SCORE_ROOT=$SCORE_ROOT WORLD_SIZE=1 \
            python /workspace/benchmarks/API-Bank/generate_batch.py \
            --model_paths "$FINAL_CKPT" >> "$LOG_FILE" 2>&1 || true

        # Run evaluate and parse output
        MANGLED=$(echo "$FINAL_CKPT" | sed 's|/|_|g')
        EVAL_LINE=$(cd /workspace/benchmarks/API-Bank && \
            PATH_TO_YOUR_SCORE_ROOT=$SCORE_ROOT python evaluate.py \
            --model_paths "$FINAL_CKPT" 2>/dev/null | grep "$MANGLED" | head -1)

        OVERALL=$(echo "$EVAL_LINE" | grep -oP '[\d.]+(?=\\%)' | sed -n '1p' || echo "0.00")
        LV1=$(echo "$EVAL_LINE" | grep -oP '[\d.]+(?=\\%)' | sed -n '2p' || echo "0.00")
        LV2=$(echo "$EVAL_LINE" | grep -oP '[\d.]+(?=\\%)' | sed -n '3p' || echo "0.00")
        LV3=$(echo "$EVAL_LINE" | grep -oP '[\d.]+(?=\\%)' | sed -n '4p' || echo "0.00")

        OVERALL=${OVERALL:-0.00}
        LV1=${LV1:-0.00}; LV2=${LV2:-0.00}; LV3=${LV3:-0.00}

        echo "${RUN_NAME},${LR},${MAX_LEN},${EPOCHS},${BATCH},${OVERALL},${LV1},${LV2},${LV3},,,,,,,0.000" >> "$RESULTS_CSV"
        echo "  [GPU $gpu_id DONE] overall=${OVERALL}% lv1=${LV1}% lv2=${LV2}% lv3=${LV3}%"
    done
}

G0=("${CONFIGS[@]:0:14}")
G1=("${CONFIGS[@]:14:14}")
G2=("${CONFIGS[@]:28:13}")
G3=("${CONFIGS[@]:41:13}")

echo "Starting 4 parallel workers (3B model)..."
echo "============================================================"

run_worker 0 "${G0[@]}" &
run_worker 1 "${G1[@]}" &
run_worker 2 "${G2[@]}" &
run_worker 3 "${G3[@]}" &

wait

echo "============================================================"
echo "Sweep complete. Results: $RESULTS_CSV"
echo "SWEEP_3B_DONE"
