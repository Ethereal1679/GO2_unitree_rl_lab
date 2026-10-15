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
    --num_envs 2048 \
    --run_name "g1_grad_VAE-remake-holes_depth:0.2-track_lin_vel_xy:2.5-track_ang_vel_z:2.0-alive:1.0-train_spacial" \
    --max_iterations 99999999 \
    --headless \
    # --resume \
    # --load_run 2026-10-11_14-35-30_cnn-vae \

