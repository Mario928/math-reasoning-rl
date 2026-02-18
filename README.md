# Math Reasoning with Reinforcement Learning

Training language models to solve math problems using supervised fine-tuning and reinforcement learning with PPO.

## Overview

This project fine-tunes **Qwen 2.5 0.5B** on the **GSM8K** dataset (grade school math problems) using:
- Supervised fine-tuning (SFT) for initial task adaptation
- Proximal Policy Optimization (PPO) for reward-based improvement
- Rule-based rewards for correct answer verification

Built with [VERL](https://github.com/volcengine/verl) for efficient RL training.

## Requirements

- Docker with NVIDIA Container Toolkit
- 1x H100 GPU (or similar GPU with 24GB+ memory)

## Quick Start

Start the containers:
```bash
docker compose up -d
docker compose ps  # verify all services are running
```

This launches three services:
- **verl**: RL training container (auto-installs VERL and fixes numpy on startup)
- **jupyter**: Interactive notebook environment for SFT (port 8888)
- **mlflow**: Experiment tracking dashboard (port 5000)

## Training Pipeline

### 1. Supervised Fine-Tuning

Open Jupyter at `http://<server-ip>:8888` and run `sft_training.ipynb`.

This fine-tunes the base model on GSM8K question-answer pairs and saves the checkpoint to the shared models volume at `/app/models/sft_qwen`.

### 2. Preprocess GSM8K for RL

```bash
docker exec verl python3 /tmp/verl/examples/data_preprocess/gsm8k.py --local_save_dir /app/data/gsm8k
```

### 3. Reinforcement Learning (PPO)

```bash
# Run in foreground (see logs directly)
docker exec verl bash /workspace/run_ppo.sh

# Or run in background
docker exec -d verl bash -c 'bash /workspace/run_ppo.sh > /tmp/ppo.log 2>&1'

# Check logs
docker exec verl tail -50 /tmp/ppo.log
```

The RL phase uses the SFT checkpoint and optimizes for correct answer generation using rule-based rewards (1.0 for correct, 0.1 for incorrect, 0 for no answer).

### 4. Merge Checkpoint (after training)

```bash
docker exec verl python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir checkpoints/verl_gsm8k/ppo_qwen_0.5b/global_step_<N>/actor \
    --target_dir /app/models/ppo_qwen
```

## Monitoring

View training metrics and loss curves at `http://<server-ip>:5000` (MLflow UI).

## Architecture

- **Shared volumes**: HuggingFace cache and model checkpoints are shared between containers
- **GPU passthrough**: Both training containers have full GPU access via NVIDIA runtime
- **Increased shared memory**: 32GB shm prevents PyTorch multiprocessing crashes
- **Auto-setup**: The verl container automatically clones VERL source and fixes numpy on startup
