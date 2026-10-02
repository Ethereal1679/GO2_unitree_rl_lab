#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"


python "${SCRIPT_DIR}/py/play.py" \
    --task go2_att \
    --num_envs 128 \
    --load_run 2026-10-01_02-40-39_resume-modify_curriculum_wo \
    --attention_enable_viz \
    # --gap_gas_enable_viz
# Add --gap_gas_enable_viz above to show the transparent green gap volumes.
