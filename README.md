# Math Reasoning with Reinforcement Learning

Training language models to solve math problems using supervised fine-tuning and reinforcement learning with PPO.

## Overview

This project fine-tunes **Qwen 2.5 0.5B** on the **GSM8K** dataset (grade school math problems) using:
- Supervised fine-tuning (SFT) for initial task adaptation
- Proximal Policy Optimization (PPO) for reward-based improvement
- Rule-based rewards for correct answer verification

Built with [VERL](https://github.com/volcengine/verl) for efficient RL training.

## Requirements

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- 1x H100 GPU (or similar GPU with 24GB+ memory)

## Quick Start

Clone the repo and start containers:
```bash
git clone https://github.com/Mario928/math-reasoning-rl.git
cd math-reasoning-rl
docker compose up -d
```

Wait ~2 minutes for the verl container to finish setup (it auto-installs VERL and fixes dependencies), then verify:
```bash
docker compose ps                    # all 3 services should be running
docker exec verl python3 -c "import verl; print(verl.__version__)"  # should print version
```

Services:
- **verl**: RL training container (auto-installs VERL on startup)
- **jupyter**: Interactive notebook for SFT — `http://<server-ip>:8888`
- **mlflow**: Experiment tracking — `http://<server-ip>:5000`

## Training Pipeline

### 1. Supervised Fine-Tuning

Open Jupyter at `http://<server-ip>:8888` and run `notebooks/sft_training.ipynb`.

This fine-tunes the base Qwen 2.5 0.5B model on GSM8K question-answer pairs and saves the checkpoint to `/app/models/sft_qwen` (shared volume between containers). The Jupyter container pins `transformers==4.57.1` to match the VERL container, ensuring tokenizer compatibility.

### 2. Preprocess GSM8K for RL

```bash
docker exec verl python3 /tmp/verl/examples/data_preprocess/gsm8k.py --local_save_dir /app/data/gsm8k
```

### 3. Reinforcement Learning (PPO)

```bash
# Run in background
docker exec -d verl bash -c 'bash /workspace/run_ppo.sh > /tmp/ppo.log 2>&1'

# Check progress
docker exec verl tail -50 /tmp/ppo.log
```

Training runs 15 epochs (~435 steps, ~3-4 hours on H100). The RL phase uses the SFT checkpoint and optimizes for correct answer generation using rule-based rewards (1.0 for correct, 0.1 for incorrect, 0 for no answer).

### 4. Merge Checkpoint (after training)

```bash
docker exec verl python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /home/dpsk_a2a/DeepEP/checkpoints/verl_gsm8k/ppo_qwen_0.5b/global_step_<N>/actor \
    --target_dir /app/models/ppo_qwen
```

Replace `<N>` with the latest checkpoint step number.

## Monitoring

View training metrics and loss curves at `http://<server-ip>:5000` (MLflow UI).

## Architecture

- **Shared volumes**: HuggingFace cache and model checkpoints shared between containers
- **GPU passthrough**: Both training containers have full GPU access via NVIDIA runtime
- **Increased shared memory**: 32GB shm prevents PyTorch multiprocessing crashes
- **Auto-setup**: The verl container automatically clones VERL source and fixes numpy on startup
