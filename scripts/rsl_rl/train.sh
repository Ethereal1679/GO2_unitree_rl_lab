#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ==注意力机制==
python "${SCRIPT_DIR}/py/train.py" \
    --task go2_att \
    --logger tensorboard \
    --num_envs 2048 \
    --run_name "add_gas_box-add_depth_curriculum-resume" \
    --max_iterations 99999999 \
    --headless \
    --resume \
    --load_run 2026-09-27_18-14-32_add_gas_box-add_depth_curriculum \




# ==测试雷达部署==
# python "${SCRIPT_DIR}/py/train.py" \
#     --task go2_lidar \
#     --logger tensorboard \
#     --num_envs 4096 \
#     --run_name "go2_lidar_test" \
#     --max_iterations 99999999 \
#     --headless

