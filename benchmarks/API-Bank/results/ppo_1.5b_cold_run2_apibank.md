# PPO 1.5B Cold Start — Run 2 (API-Bank)

Re-trained from scratch with same paper config (Qwen2.5-1.5B-Instruct cold start, 15 epochs, train_batch_size=512, ppo_mini=128, lr=1e-6, kl_coef=0.001) on 4xH100 with mlflow logging.

## Results
| Metric | Value |
|---|---|
| Overall | **52.09%** |
| L1 | 58.15% |
| L2 | 40.3% |
| L3 | 39.69% |

## Comparison
| Source | Overall |
|---|---|
| Paper | 40.54% |
| Run 1 | 56.28% |
| **Run 2** | **52.09%** |

Both runs outperform paper's 40.54% — confirms the +15pt anomaly was not a single-seed fluke.
Run 2 (52.09%) is closer to paper than Run 1 (56.28%) but both still significantly beat it (+11.55pt and +15.74pt respectively).
