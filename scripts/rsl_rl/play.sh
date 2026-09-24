#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"


python "${SCRIPT_DIR}/py/play.py" \
    --task go2_att \
    --num_envs 32 \
    --load_run 2026-09-24_15-20-40_go2_attention_train2 \
    --attention_enable_viz
