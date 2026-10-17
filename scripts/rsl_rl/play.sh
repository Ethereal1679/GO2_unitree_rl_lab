#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOADRUN_NAME="2026-10-02_23-34-35_no_resume-ABLATION-rm_gap_penetration_penalty-M"

# OpenCV cv.imshow requires a desktop Isaac Sim session.
export HEADLESS=0

# python "${SCRIPT_DIR}/py/play.py" \
#     --task go2_lidar \
#     --num_envs 128 \
#     --load_run ${LOADRUN_NAME} \
#     --attention_enable_viz \
    # --checkpoint /home/ab123456/文档/GO2_unitree_rl_lab/logs/rsl_rl/go2_att/${LOADRUN_NAME}/model_15000.pt
#     # --gap_gas_enable_viz
# # Add --gap_gas_enable_viz above to show the transparent green gap volumes.



# --height_scan_gradient_viz \
# --gap_gas_enable_viz \
python "${SCRIPT_DIR}/py/play.py" \
    --task g1_lidar \
    --num_envs 64 \
    --load_run 2026-10-17_18-51-44_add_stair \
    --gap_gas_enable_viz \
    # --grad_cnn_map_viz \
    # --height_scan_gradient_viz

