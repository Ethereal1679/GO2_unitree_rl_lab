#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="2026-09-22_17-15-47_go2_attention"

python "${SCRIPT_DIR}/play.py" \
    --task go2_att \
    --num_envs 32 \
    --load_run "${RUN_NAME}" \
    --attention_weights \
