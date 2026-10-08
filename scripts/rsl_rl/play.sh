#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOADRUN_NAME="2026-10-02_23-34-35_no_resume-ABLATION-rm_gap_penetration_penalty-M"

python "${SCRIPT_DIR}/py/play.py" \
    --task go2_att \
    --num_envs 128 \
    --load_run ${LOADRUN_NAME} \
    --attention_enable_viz \
    --checkpoint /home/ab123456/文档/GO2_unitree_rl_lab/logs/rsl_rl/go2_att/${LOADRUN_NAME}/model_15000.pt
    # --gap_gas_enable_viz
# Add --gap_gas_enable_viz above to show the transparent green gap volumes.
