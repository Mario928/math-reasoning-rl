#!/bin/bash
# ============================================================
# API-Bank Eval for all 54 SFT Sweep Checkpoints
# Runs 4 parallel workers (1 per GPU), generate then evaluate
# Saves full results to eval_results.csv
# ============================================================

CHECKPOINT_DIR="/workspace/sweep_checkpoints/qwen2.5-1.5b"
APIBANK_DIR="/workspace/benchmarks/API-Bank"
RESULTS_DIR="/workspace/sweep_results/qwen2.5-1.5b"
CSV_OUT="${RESULTS_DIR}/eval_results.csv"

# Collect all checkpoint dirs
CKPTS=()
for ckpt in "$CHECKPOINT_DIR"/sft_*/; do
    [ -d "$ckpt" ] && CKPTS+=("${ckpt%/}")
done

echo "Found ${#CKPTS[@]} checkpoints to evaluate"

# Write CSV header
echo "run_name,lr,max_length,epochs,batch_size,overall_acc,lv1_acc,lv2_acc,lv3_acc,correct_lv1,correct_lv2,correct_lv3,total_lv1,total_lv2,total_lv3,val_loss" > "$CSV_OUT"

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

        # Check if already done (score.json exists)
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

# Run evaluate.py for all checkpoints (scores + leaderboard.json)
CKPT_LIST=$(IFS=,; echo "${CKPTS[*]}")
cd "$APIBANK_DIR"
python evaluate.py --model_paths "$CKPT_LIST"

echo ""
echo "============================================================"
echo "Building eval_results.csv with all metrics..."
echo "============================================================"

# Parse leaderboard.json + val loss from logs -> write full CSV
python3 - <<'PYEOF'
import json, os, re, csv

APIBANK_DIR = "/workspace/benchmarks/API-Bank"
RESULTS_DIR = "/workspace/sweep_results/qwen2.5-1.5b"
CHECKPOINT_DIR = "/workspace/sweep_checkpoints/qwen2.5-1.5b"
CSV_OUT = os.path.join(RESULTS_DIR, "eval_results.csv")

leaderboard_path = os.path.join(APIBANK_DIR, "PATH_TO_YOUR_SCORE_ROOT", "leaderboard.json")
leaderboard = json.load(open(leaderboard_path)) if os.path.exists(leaderboard_path) else {}

rows = []
for run_name in sorted(os.listdir(CHECKPOINT_DIR)):
    if not run_name.startswith("sft_"):
        continue

    # Parse hyperparams from run name: sft_lr{LR}_len{MAX_LEN}_ep{EPOCHS}_bs{BATCH}
    m = re.match(r"sft_lr(.+)_len(\d+)_ep(\d+)_bs(\d+)", run_name)
    if not m:
        continue
    lr, max_len, epochs, batch = m.group(1), m.group(2), m.group(3), m.group(4)

    # Get API-Bank scores from leaderboard
    scores = leaderboard.get(run_name, {})
    overall = scores.get("overall_acc", "")
    lv1 = scores.get("lv1_acc", "")
    lv2 = scores.get("lv2_acc", "")
    lv3 = scores.get("lv3_acc", "")
    c1 = scores.get("correct_lv1", "")
    c2 = scores.get("correct_lv2", "")
    c3 = scores.get("correct_lv3", "")
    t1 = scores.get("total_lv1", "")
    t2 = scores.get("total_lv2", "")
    t3 = scores.get("total_lv3", "")

    # Get final val loss from sweep log
    log_path = os.path.join(RESULTS_DIR, f"{run_name}.log")
    val_loss = ""
    if os.path.exists(log_path):
        for line in open(log_path):
            # fsdp_sft_trainer logs: val/loss: X.XXXX
            m2 = re.search(r"val/loss[:\s]+([0-9.]+)", line)
            if m2:
                val_loss = m2.group(1)  # keeps updating -> last match = final epoch

    rows.append([run_name, lr, max_len, epochs, batch,
                 overall, lv1, lv2, lv3,
                 c1, c2, c3, t1, t2, t3, val_loss])

# Sort by overall_acc descending
rows.sort(key=lambda r: float(r[5]) if r[5] != "" else -1, reverse=True)

with open(CSV_OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_name","lr","max_length","epochs","batch_size",
                "overall_acc","lv1_acc","lv2_acc","lv3_acc",
                "correct_lv1","correct_lv2","correct_lv3",
                "total_lv1","total_lv2","total_lv3","val_loss"])
    w.writerows(rows)

print(f"Saved {len(rows)} rows to {CSV_OUT}")
print("\nTop 10 by overall API-Bank accuracy:")
print(f"{'run_name':<45} {'overall':>8} {'lv1':>7} {'lv2':>7} {'lv3':>7} {'val_loss':>10}")
print("-" * 90)
for r in rows[:10]:
    print(f"{r[0]:<45} {str(r[5]):>8} {str(r[6]):>7} {str(r[7]):>7} {str(r[8]):>7} {str(r[15]):>10}")
PYEOF

echo ""
echo "Done. Results at: $CSV_OUT"
echo "Leaderboard at:   $APIBANK_DIR/PATH_TO_YOUR_SCORE_ROOT/leaderboard.json"
