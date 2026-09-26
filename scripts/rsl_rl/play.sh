#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"


python "${SCRIPT_DIR}/py/play.py" \
    --task go2_att \
    --num_envs 128 \
    --load_run 2026-09-26_16-32-43_no_resume-add_stuck_penalty-std-terrain_easy \
    --attention_enable_viz
