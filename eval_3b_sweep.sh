#!/bin/bash
# Post-sweep eval for 3B SFT checkpoints
# Runs generate_batch.py + evaluate.py on all completed checkpoints
# Output: /workspace/sweep_results/qwen2.5-3b/eval_results.csv (same format as 1.5B)

CHECKPOINT_DIR="/workspace/sweep_checkpoints/qwen2.5-3b"
RESULTS_DIR="/workspace/sweep_results/qwen2.5-3b"
SCORE_ROOT="${RESULTS_DIR}/apibank_scores"
RESULTS_CSV="${RESULTS_DIR}/eval_results.csv"

mkdir -p "$SCORE_ROOT"

echo "run_name,lr,max_length,epochs,batch_size,overall_acc,lv1_acc,lv2_acc,lv3_acc,correct_lv1,correct_lv2,correct_lv3,total_lv1,total_lv2,total_lv3,val_loss" > "$RESULTS_CSV"

cd /workspace/benchmarks/API-Bank

# Collect all final checkpoints
CKPTS=()
for run_dir in "$CHECKPOINT_DIR"/*/; do
    final_step=$(ls "$run_dir" | grep global_step | sort -t_ -k3 -n | tail -1)
    if [ -n "$final_step" ]; then
        CKPTS+=("$run_dir$final_step")
    fi
done

echo "Found ${#CKPTS[@]} checkpoints to evaluate"

# Split into 4 GPU groups
total=${#CKPTS[@]}
per_gpu=$(( (total + 3) / 4 ))

eval_worker() {
    local gpu_id=$1
    shift
    local my_ckpts=("$@")

    export CUDA_VISIBLE_DEVICES=$gpu_id

    for ckpt in "${my_ckpts[@]}"; do
        run_name=$(basename "$(dirname "$ckpt")")
        echo "[GPU $gpu_id] Evaluating $run_name..."

        # Skip if result already exists
        mangled=$(echo "$ckpt" | sed 's|/|_|g')
        if [ -f "${SCORE_ROOT}/${mangled}/result.json" ]; then
            echo "  [GPU $gpu_id SKIP] result exists"
        else
            PATH_TO_YOUR_SCORE_ROOT=$SCORE_ROOT WORLD_SIZE=1 \
                python generate_batch.py --model_paths "$ckpt" 2>&1 | tail -3
        fi

        # Parse val_loss from training log
        log_file="${RESULTS_DIR}/${run_name}.log"
        val_loss=$(grep 'val/loss' "$log_file" 2>/dev/null | tail -1 | grep -oP '[\d.]+$' || echo "0.000")

        # Parse config from run_name: sft_lr{LR}_len{MAX_LEN}_ep{EPOCHS}_bs{BATCH}
        LR=$(echo "$run_name" | grep -oP '(?<=lr)[\de\-]+')
        MAX_LEN=$(echo "$run_name" | grep -oP '(?<=len)\d+')
        EPOCHS=$(echo "$run_name" | grep -oP '(?<=ep)\d+')
        BATCH=$(echo "$run_name" | grep -oP '(?<=bs)\d+')

        # Run evaluate and capture numbers
        EVAL_OUT=$(PATH_TO_YOUR_SCORE_ROOT=$SCORE_ROOT python evaluate.py \
            --model_paths "$ckpt" 2>/dev/null | grep "$mangled" | head -1)

        OVERALL=$(echo "$EVAL_OUT" | grep -oP '[\d.]+(?=\\%)' | sed -n '1p')
        LV1=$(echo "$EVAL_OUT" | grep -oP '[\d.]+(?=\\%)' | sed -n '2p')
        LV2=$(echo "$EVAL_OUT" | grep -oP '[\d.]+(?=\\%)' | sed -n '3p')
        LV3=$(echo "$EVAL_OUT" | grep -oP '[\d.]+(?=\\%)' | sed -n '4p')

        # correct counts from eval output: & N & N & N
        C1=$(echo "$EVAL_OUT" | grep -oP '(?<=& )\d+' | sed -n '1p' || echo "0")
        C2=$(echo "$EVAL_OUT" | grep -oP '(?<=& )\d+' | sed -n '2p' || echo "0")
        C3=$(echo "$EVAL_OUT" | grep -oP '(?<=& )\d+' | sed -n '3p' || echo "0")

        OVERALL=${OVERALL:-0.00}
        LV1=${LV1:-0.00}; LV2=${LV2:-0.00}; LV3=${LV3:-0.00}

        echo "${run_name},${LR},${MAX_LEN},${EPOCHS},${BATCH},${OVERALL},${LV1},${LV2},${LV3},${C1},${C2},${C3},399,67,131,${val_loss}" >> "$RESULTS_CSV"
        echo "  [GPU $gpu_id DONE] overall=${OVERALL}%"
    done
}

# Split checkpoints into 4 groups
G0=("${CKPTS[@]:0:$per_gpu}")
G1=("${CKPTS[@]:$per_gpu:$per_gpu}")
G2=("${CKPTS[@]:$((per_gpu*2)):$per_gpu}")
G3=("${CKPTS[@]:$((per_gpu*3))}")

echo "Starting 4 parallel eval workers..."
eval_worker 0 "${G0[@]}" &
eval_worker 1 "${G1[@]}" &
eval_worker 2 "${G2[@]}" &
eval_worker 3 "${G3[@]}" &

wait

# Sort CSV by overall_acc descending
echo "Sorting results..."
head -1 "$RESULTS_CSV" > /tmp/header.csv
tail -n +2 "$RESULTS_CSV" | sort -t',' -k6 -rn >> /tmp/header.csv
mv /tmp/header.csv "$RESULTS_CSV"

echo "EVAL_DONE. Results: $RESULTS_CSV"
cat "$RESULTS_CSV" | head -10
