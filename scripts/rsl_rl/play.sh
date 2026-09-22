#!/usr/bin/env bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUN_NAME="2026-09-20_23-31-21_go2_lidar-add_vae-trimesh_terrain"

python "${SCRIPT_DIR}/play.py" \
    --task go2_lidar \
    --num_envs 32 \
    --load_run "${RUN_NAME}"
