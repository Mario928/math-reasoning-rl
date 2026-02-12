#!/bin/bash

export N_GPUS=4
export MLFLOW_TRACKING_URI=http://mlflow:5000

python3 -m verl.trainer.main_ppo \
    data.train_files=/app/data/gsm8k/train.parquet \
    data.val_files=/app/data/gsm8k/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    actor_rollout_ref.model.path=/app/models/sft_qwen \
    critic.model.path=/app/models/sft_qwen \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 \
    trainer.logger='["mlflow"]' \
    trainer.project_name=verl_gsm8k \
    trainer.experiment_name=ppo_qwen_0.5b
