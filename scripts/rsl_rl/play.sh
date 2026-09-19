#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="2026-08-04_23-34-42_rough_terrain_4096_envs"

python "${SCRIPT_DIR}/play.py" \
    --task g1_amp \
    --num_envs 256 \
    --seed 114514 \
    --load_run "${RUN_NAME}"
