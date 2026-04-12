#!/bin/bash

echo "[Sat Apr 11 06:19:05 EDT 2026] GRPO 1.5B START" >> /tmp/pipeline.log
sudo docker exec verl bash -c 'cd /workspace && bash train_grpo.sh > /tmp/grpo.log 2>&1'
[ 0 -eq 0 ] && echo "[Sat Apr 11 06:19:05 EDT 2026] GRPO 1.5B DONE" >> /tmp/pipeline.log || echo "[Sat Apr 11 06:19:05 EDT 2026] GRPO 1.5B FAILED" >> /tmp/pipeline.log

echo "[Sat Apr 11 06:19:05 EDT 2026] PPO 1.5B START" >> /tmp/pipeline.log
sudo docker exec verl bash -c 'cd /workspace && bash train_ppo.sh > /tmp/ppo.log 2>&1'
[ 0 -eq 0 ] && echo "[Sat Apr 11 06:19:05 EDT 2026] PPO 1.5B DONE" >> /tmp/pipeline.log || echo "[Sat Apr 11 06:19:05 EDT 2026] PPO 1.5B FAILED" >> /tmp/pipeline.log

echo "[Sat Apr 11 06:19:05 EDT 2026] GRPO 3B START" >> /tmp/pipeline.log
sudo docker exec verl bash -c 'cd /workspace && bash train_grpo_3b.sh > /tmp/grpo3b.log 2>&1'
[ 0 -eq 0 ] && echo "[Sat Apr 11 06:19:05 EDT 2026] GRPO 3B DONE" >> /tmp/pipeline.log || echo "[Sat Apr 11 06:19:05 EDT 2026] GRPO 3B FAILED" >> /tmp/pipeline.log

echo "[Sat Apr 11 06:19:05 EDT 2026] PPO 3B START" >> /tmp/pipeline.log
sudo docker exec verl bash -c 'cd /workspace && bash train_ppo_3b.sh > /tmp/ppo3b.log 2>&1'
[ 0 -eq 0 ] && echo "[Sat Apr 11 06:19:05 EDT 2026] PPO 3B DONE" >> /tmp/pipeline.log || echo "[Sat Apr 11 06:19:05 EDT 2026] PPO 3B FAILED" >> /tmp/pipeline.log

echo "[Sat Apr 11 06:19:05 EDT 2026] ALL DONE" >> /tmp/pipeline.log
