#!/usr/bin/env bash
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"


python "${SCRIPT_DIR}/py/play.py" \
    --task go2_att \
    --num_envs 128 \
    --load_run 2026-09-26_22-02-03_16dim_4head \
    --attention_enable_viz
