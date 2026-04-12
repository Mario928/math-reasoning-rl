#!/bin/bash
# ============================================================
# API-Bank Eval for all 54 SFT Sweep Checkpoints
# Runs 4 parallel workers (1 per GPU), generate then evaluate
# ============================================================

CHECKPOINT_DIR="/workspace/sweep_checkpoints/qwen2.5-1.5b"
APIBANK_DIR="/workspace/benchmarks/API-Bank"

# Collect all checkpoint dirs (each is one completed run)
CKPTS=()
for ckpt in "$CHECKPOINT_DIR"/sft_*/; do
    [ -d "$ckpt" ] && CKPTS+=("${ckpt%/}")
done

echo "Found ${#CKPTS[@]} checkpoints to evaluate"

# Split into 4 groups for parallel GPU workers
total=${#CKPTS[@]}
g0_end=$(( (total + 3) / 4 ))

G0=("${CKPTS[@]:0:$g0_end}")
G1=("${CKPTS[@]:$g0_end:$g0_end}")
G2=("${CKPTS[@]:$(( g0_end * 2 )):$g0_end}")
G3=("${CKPTS[@]:$(( g0_end * 3 ))}")

run_worker() {
    local gpu_id=$1
    shift
    local my_ckpts=("$@")

    export CUDA_VISIBLE_DEVICES=$gpu_id

    for ckpt_path in "${my_ckpts[@]}"; do
        run_name=$(basename "$ckpt_path")
        echo "[GPU $gpu_id] Evaluating: $run_name"

        # Check if already done (score.json exists for this model)
        score_file="$APIBANK_DIR/PATH_TO_YOUR_SCORE_ROOT/${run_name}/score.json"
        if [ -f "$score_file" ]; then
            echo "  [GPU $gpu_id SKIP] already evaluated: $run_name"
            continue
        fi

        # generate.py must run from API-Bank dir (reads ./level-*.json relative paths)
        cd "$APIBANK_DIR"
        python generate.py --model_paths "$ckpt_path" || true
        echo "  [GPU $gpu_id] generate done: $run_name"
    done

    echo "[GPU $gpu_id] All done."
}

echo "Starting 4 parallel generate workers..."
echo "============================================================"

run_worker 0 "${G0[@]}" &
run_worker 1 "${G1[@]}" &
run_worker 2 "${G2[@]}" &
run_worker 3 "${G3[@]}" &

wait

echo ""
echo "============================================================"
echo "All generate done. Running evaluate + leaderboard..."
echo "============================================================"

# evaluate.py scores all result.json files and prints ranked leaderboard
CKPT_LIST=$(IFS=,; echo "${CKPTS[*]}")
cd "$APIBANK_DIR"
python evaluate.py --model_paths "$CKPT_LIST"

echo ""
echo "Done. Leaderboard at: $APIBANK_DIR/PATH_TO_YOUR_SCORE_ROOT/leaderboard.json"
