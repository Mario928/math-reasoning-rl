#!/bin/bash
# PPO training script for Qwen 2.5 0.5B on GSM8K
# Based on official VERL quickstart: https://verl.readthedocs.io/en/latest/start/quickstart.html

set -x

export MLFLOW_TRACKING_URI=http://mlflow:5000

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=/app/data/gsm8k/train.parquet \
    data.val_files=/app/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=/app/models/sft_qwen \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    critic.optim.lr=1e-5 \
    critic.model.path=/app/models/sft_qwen \
    critic.ppo_micro_batch_size_per_gpu=4 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.val_before_train=False \
    trainer.logger='["console","mlflow"]' \
    trainer.project_name=verl_gsm8k \
    trainer.experiment_name=ppo_qwen_0.5b \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15
