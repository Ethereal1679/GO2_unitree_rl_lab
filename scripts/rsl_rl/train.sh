#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ==注意力机制==
python "${SCRIPT_DIR}/py/train.py" \
    --task go2_att \
    --logger tensorboard \
    --num_envs 2048 \
    --run_name "go2_attention-no_resume-remake_policy" \
    --max_iterations 99999999 \
    --headless \
    # --resume \
    # --load_run 2026-09-24_03-51-08_go2_attention_train2 \



# ==测试雷达部署==
# python "${SCRIPT_DIR}/py/train.py" \
#     --task go2_lidar \
#     --logger tensorboard \
#     --num_envs 4096 \
#     --run_name "go2_lidar_test" \
#     --max_iterations 99999999 \
#     --headless

