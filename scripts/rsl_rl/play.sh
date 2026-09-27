#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"


python "${SCRIPT_DIR}/py/play.py" \
    --task go2_att \
    --num_envs 128 \
    --load_run 2026-09-27_19-02-42_add_gas_box-add_depth_curriculum-resume \
    --attention_enable_viz \
    # --gap_gas_enable_viz
# Add --gap_gas_enable_viz above to show the transparent green gap volumes.
