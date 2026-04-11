export CUDA_VISIBLE_DEVICES=0,1,2,3
export N_GPUS=4
export ROLLOUT_TP_SIZE=1
export VLLM_ATTENTION_BACKEND=XFORMERS

export WITHLENGTH=0
export REFINEDREWARD=0
export COARSEREWARD=0
export STRICTMATCH=0
export CORRECTMAX1=0
export MAX1STEP30MAX3=0
export SCHEDULEREWARD=0
export SCHEDULELENGTH=0

export DATA_DIR="./dataset/rlla_4k"
export BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
export EXPERIMENT_NAME="/app/models/toolrl-grpo-qwen-3b"
bash ./examples/grpo_trainer/run_grpo.sh
