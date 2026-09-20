#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/train.py" \
    --task go2_lidar \
    --headless \
    --logger tensorboard \
    --num_envs 4096 \
    --run_name "go2_lidar-add_vae" \
    --max_iterations 99999999
    # --resume \
    # --load_run "2026-08-04_23-34-42_rough_terrain_4096_envs" \




# python "${SCRIPT_DIR}/train.py" \
#     --task go2_lidar \
#     --logger tensorboard \
#     --num_envs 64 \
#     --run_name "test"


