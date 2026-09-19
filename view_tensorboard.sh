#!/bin/bash

(sleep 2; xdg-open http://localhost:6006 >/dev/null 2>&1) &
exec tensorboard \
    --logdir /home/ab123456/文档/GO2_unitree_rl_lab/logs/rsl_rl/go2_lidar/ \
    # --port 6006