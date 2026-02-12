# Math Reasoning with Reinforcement Learning

Training language models to solve math problems using supervised fine-tuning and reinforcement learning with PPO.

## Overview

This project fine-tunes **Qwen 0.5B** on the **GSM8K** dataset (grade school math problems) using:
- Supervised fine-tuning (SFT) for initial task adaptation
- Proximal Policy Optimization (PPO) for reward-based improvement
- Function-based rewards for correct answer verification

Built with [VERL](https://github.com/volcengine/verl) for efficient RL training.

## Requirements

- Docker with NVIDIA Container Toolkit
- 4x H100 GPUs (or similar high-memory GPUs)

## Quick Start

Start the containers:
```bash
docker compose up -d
docker compose ps  # verify all services are running
```

This launches three services:
- **jupyter**: Interactive notebook environment for SFT
- **verl**: RL training container
- **mlflow**: Experiment tracking dashboard

## Training Pipeline

### 1. Supervised Fine-Tuning

Open Jupyter at `http://<server-ip>:8888` and run `sft_training.ipynb`

This fine-tunes the base model on GSM8K problems and saves the checkpoint to the shared models volume.

### 2. Reinforcement Learning

Preprocess the dataset:
```bash
docker compose exec verl python3 examples/data_preprocess/gsm8k.py --local_save_dir /app/data/gsm8k
```

Run PPO training:
```bash
docker compose exec verl bash /workspace/run_ppo.sh
```

The RL phase uses the SFT checkpoint and optimizes for correct answer generation using rule-based rewards.

## Monitoring

View training metrics and loss curves at `http://<server-ip>:5000` (MLflow UI)

## Results

Training combines supervised fine-tuning (learning format from examples) with reinforcement learning (optimizing for correct answers via reward feedback). The model improves from baseline accuracy to better math reasoning performance over the training process.

## Architecture

- **Shared volumes**: HuggingFace cache and model checkpoints are shared between containers to avoid redundant downloads
- **GPU passthrough**: Both training containers have full GPU access
- **Increased shared memory**: 32GB shm prevents PyTorch multiprocessing crashes
