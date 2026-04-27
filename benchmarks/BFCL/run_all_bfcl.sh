#!/bin/bash
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

CATEGORIES="simple_python,simple_java,simple_javascript,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel,live_parallel_multiple,live_irrelevance,live_relevance,multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context,memory_kv,memory_vector,memory_rec_sum"

declare -A MODEL_PATHS
MODEL_PATHS["toolrl-grpo-qwen-1.5b"]="/app/models/toolrl-grpo-qwen-1.5b/actor/global_step_90"
MODEL_PATHS["toolrl-grpo-coarse-qwen-1.5b"]="/app/models/toolrl-grpo-coarse-qwen-1.5b/actor/global_step_90"
MODEL_PATHS["toolrl-grpo-qwen-3b"]="/app/models/toolrl-grpo-qwen-3b/actor/global_step_90"
MODEL_PATHS["toolrl-grpo-sft400-qwen-1.5b"]="/app/models/toolrl-grpo-sft400-qwen-1.5b/actor/global_step_90"
MODEL_PATHS["toolrl-grpo-sft400-qwen-3b"]="/app/models/toolrl-grpo-sft400-qwen-3b/actor/global_step_90"
MODEL_PATHS["toolrl-ppo-qwen-1.5b"]="/app/models/toolrl-ppo-qwen-1.5b/actor/global_step_90"
MODEL_PATHS["toolrl-ppo-qwen-3b"]="/app/models/toolrl-ppo-qwen-3b/actor/global_step_90"
MODEL_PATHS["toolrl-ppo-sft400-qwen-1.5b"]="/app/models/toolrl-ppo-sft400-qwen-1.5b/actor/global_step_90"
MODEL_PATHS["toolrl-ppo-sft400-qwen-3b"]="/app/models/toolrl-ppo-sft400-qwen-3b/actor/global_step_90"
MODEL_PATHS["toolrl-ppo-qwen-1.5b-run2"]="/app/models/toolrl-ppo-qwen-1.5b-run2/actor/global_step_90"

ORDER=(
    "toolrl-grpo-qwen-1.5b"
    "toolrl-grpo-coarse-qwen-1.5b"
    "toolrl-grpo-qwen-3b"
    "toolrl-grpo-sft400-qwen-1.5b"
    "toolrl-grpo-sft400-qwen-3b"
    "toolrl-ppo-qwen-1.5b"
    "toolrl-ppo-qwen-3b"
    "toolrl-ppo-sft400-qwen-1.5b"
    "toolrl-ppo-sft400-qwen-3b"
    "toolrl-ppo-qwen-1.5b-run2"
)

for MODEL in "${ORDER[@]}"; do
    MPATH="${MODEL_PATHS[$MODEL]}"
    LOG="/tmp/bfcl_${MODEL}.log"
    echo "========================================"
    echo "$(date): Starting $MODEL"
    echo "Path: $MPATH"

    pkill -f vllm 2>/dev/null || true
    sleep 5

    echo "$(date): Running generate for $MODEL"
    bfcl generate \
        --model "$MODEL" \
        --test-category "$CATEGORIES" \
        --backend vllm \
        --local-model-path "$MPATH" \
        --num-gpus 4 \
        --gpu-memory-utilization 0.85 \
        --allow-overwrite > "$LOG" 2>&1

    if [ $? -ne 0 ]; then
        echo "$(date): GENERATE FAILED for $MODEL — skipping evaluate, continuing to next"
        continue
    fi

    echo "$(date): Running evaluate for $MODEL"
    bfcl evaluate \
        --model "$MODEL" \
        --test-category "$CATEGORIES" >> "$LOG" 2>&1

    echo "$(date): DONE $MODEL"
done

echo "=== ALL BFCL RUNS COMPLETE $(date) ==="
