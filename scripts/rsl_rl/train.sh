#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# # ==注意力机制==
# python "${SCRIPT_DIR}/py/train.py" \
#     --task go2_att \
#     --logger tensorboard \
#     --num_envs 2048 \
#     --run_name "no_resume-ABLATION-rm_gap_penetration_penalty" \
#     --max_iterations 99999999 \
#     --headless \
    # --resume \
    # --load_run 2026-09-30_23-42-13_resume-heading_error_std \





# ==测试雷达部署==
# python "${SCRIPT_DIR}/py/train.py" \
#     --task go2_lidar \
#     --logger tensorboard \
#     --num_envs 4096 \
#     --run_name "go2_lidar_test" \
#     --max_iterations 99999999 \
#     --headless




# == G1 普通 ==
python "${SCRIPT_DIR}/py/train.py" \
    --task g1_lidar \
    --logger tensorboard \
    --num_envs 4096 \
    --run_name "no_resume-g1-init-heading_scale-2.0" \
    --max_iterations 99999999 \
    --headless \
    # --resume \
    # --load_run 2026-10-08_19-58-40_no_resume-g1-init \

