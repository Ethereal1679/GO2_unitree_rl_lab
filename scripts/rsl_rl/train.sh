#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/train.py" \
    --task go2_att \
    --logger tensorboard \
    --num_envs 2048 \
    --run_name "go2_attention" \
    --max_iterations 99999999 \
    --headless




# python "${SCRIPT_DIR}/train.py" \
#     --task go2_lidar \
#     --logger tensorboard \
#     --num_envs 64 \
#     --run_name "test"

