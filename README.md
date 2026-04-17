# ToolRL: Reward is All Tool Learning Needs
[**🤗 Model**](https://huggingface.co/collections/emrecanacikgoz/toolrl-680706679204ead5a6d44f58) | [**📊 Dataset**](https://github.com/qiancheng0/ToolRL/tree/main/dataset) | [**📖 Paper**](https://arxiv.org/pdf/2504.13958)

![DataPipeline](assets/reward.png)

ToolRL is the code repository for paper "ToolRL: Reward is All Tool Learning Needs".

Our code is built upon [veRL](https://github.com/volcengine/verl) and [TinyZero](https://github.com/Jiayi-Pan/TinyZero).

---

## Reproduction (NYU Advanced Project)

This branch (`toolrl-reproduction`) contains our full reproduction of the ToolRL paper for Qwen2.5-1.5B and Qwen2.5-3B, including:
- Cold start GRPO and PPO training
- SFT baseline sweep (54 configs, 1.5B and 3B)
- Warm start GRPO and PPO (SFT400 → RL)
- API-Bank evaluation

### Our Results vs Paper (API-Bank Overall Accuracy)

#### Qwen2.5-1.5B
| Model | Paper | Ours |
|-------|-------|------|
| Raw (no training) | 30.65% | 31.32% |
| SFT400 | 53.60% | 53.77% |
| SFT400 + GRPO (warm) | 61.31% | 55.95% |
| SFT400 + PPO (warm) | 57.12% | 52.60% |
| GRPO Cold Start | 63.15% | 58.96% |
| PPO Cold Start | 40.54% | 56.28% |

#### Qwen2.5-3B
| Model | Paper | Ours |
|-------|-------|------|
| Raw (no training) | 51.59% | 33.84% |
| SFT400 | 52.76% | 56.78% |
| SFT400 + GRPO (warm) | 62.48% | 54.61% |
| SFT400 + PPO (warm) | 65.16% | — |
| GRPO Cold Start | 67.00% | 62.98% |
| PPO Cold Start | 57.62% | 47.57% |

### Infrastructure
- Cold start GRPO/PPO: 4x H100 80GB (KVM, Chameleon Cloud)
- SFT sweep + warm start: 4x A100 80GB (bare metal, CHI@UC)
- Experiment tracking: MLflow
- Docker image: `nvidia/cuda:12.1.0-devel-ubuntu22.04` base, vllm==0.6.3

---

## Docker Setup (Recommended)

We provide a Dockerfile and docker-compose for a fully reproducible environment.

```bash
git clone -b toolrl-reproduction https://github.com/Mario928/math-reasoning-rl.git
cd math-reasoning-rl

# Create volumes
sudo docker volume create --name math-reasoning-rl_hf_cache
sudo docker volume create --name math-reasoning-rl_models
sudo docker volume create --name math-reasoning-rl_mlflow_data
sudo docker volume create --name math-reasoning-rl_datasets

# Build and start (MLflow at :5000, training container: verl)
sudo docker compose up -d --build
```

Download base model inside container:
```bash
sudo docker exec verl bash -c 'cd /workspace && python -c "
from huggingface_hub import snapshot_download
snapshot_download(\"Qwen/Qwen2.5-1.5B-Instruct\", local_dir=\"/app/models/Qwen2.5-1.5B-Instruct\")
snapshot_download(\"Qwen/Qwen2.5-3B-Instruct\", local_dir=\"/app/models/Qwen2.5-3B-Instruct\")
"'
```

---

## 🔍 Installation (without Docker)

```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install ray
pip install -e .
pip install flash-attn --no-build-isolation
```

---

## 📊 Dataset

Raw data is in `./dataset/rlla_4k_raw` (2K ToolACE + 1K Hammer + 1K xLAM).

Processed datasets (ready to use):
- `./dataset/rlla_4k/` — RL training data (4000 samples, `data_source/prompt/ability/reward_model/extra_info` columns)
- `./dataset/rlla_sft/` — SFT training data (380 train / 20 val, `prompt/response` columns)

To regenerate:
```bash
# RL format (for GRPO/PPO)
python dataset/rlla_4k_raw/rlla.py

# SFT format (for fsdp_sft_trainer)
python dataset/rlla_sft/make_sft_parquet.py
```

> **Warning**: `make_sft_parquet_4k.py` writes to `dataset/rlla_4k/train.parquet` and overwrites the RL-format file. Always re-run `rlla.py` before GRPO/PPO training if you ran the SFT 4k script.

---

## 🧪 Training

### Cold Start GRPO / PPO (1.5B or 3B)

Edit `train_grpo.sh` or `train_ppo.sh` to set `BASE_MODEL` and `EXPERIMENT_NAME`, then:
```bash
bash train_grpo.sh   # GRPO
bash train_ppo.sh    # PPO
```

For 3B models, set `ROLLOUT_TP_SIZE=2` in the script.

To run all 4 cold start experiments sequentially:
```bash
bash run_pipeline.sh
```

### SFT Sweep (HP search over 54 configs)

```bash
# 1.5B model, 400 samples
bash sft_sweep.sh

# 3B model, 400 samples
bash sft_sweep_3b.sh
```

Evaluate all checkpoints on API-Bank:
```bash
bash sft_eval.sh          # 1.5B
bash eval_3b_sweep.sh     # 3B
```

Best configs found:
- **1.5B**: lr=5e-5, max_len=4096, epochs=5, batch=64 → **53.77%**
- **3B**: lr=5e-5, max_len=4096, epochs=3, batch=32 → **56.78%**

### Warm Start (SFT → RL)

Set `BASE_MODEL` in `train_grpo.sh` / `train_ppo.sh` to your best SFT checkpoint, then run as above.

### Single SFT run with MLflow

```bash
python train_sft.py --model 1.5b --dataset 400
```

---

## 📈 Evaluation (API-Bank)

```bash
# Fast batched inference (recommended — ~5 min for 597 questions)
cd benchmarks/API-Bank
WORLD_SIZE=4 python generate_batch.py --model_paths /app/models/your-model/global_step_N

# Score the outputs
python evaluate.py --model_paths /app/models/your-model/global_step_N
```

Results are saved to `benchmarks/API-Bank/PATH_TO_YOUR_SCORE_ROOT/`.

---

### Reward variants
```bash
export WITHLENGTH=1        # settled length reward (Section 5.1)
export SCHEDULELENGTH=1    # dynamic length reward (Section 5.1)
export CORRECTMAX1=1       # equal max (Section 5.2)
export MAX1STEP30MAX3=1    # two stage scale (Section 5.2)
export SCHEDULEREWARD=1    # smooth dynamic scale (Section 5.2)
export REFINEDREWARD=1     # finegrained reward (Section 5.3)
export COARSEREWARD=1      # coarse reward (Section 5.3)
```

---

## 🖊️ Citation
```text
@article{qian2025toolrl,
  title={ToolRL: Reward is All Tool Learning Needs},
  author={Qian, Cheng and Acikgoz, Emre Can and He, Qi and Wang, Hongru and Chen, Xiusi and Hakkani-T{\"u}r, Dilek and Tur, Gokhan and Ji, Heng},
  journal={arXiv preprint arXiv:2504.13958},
  year={2025}
}
```
