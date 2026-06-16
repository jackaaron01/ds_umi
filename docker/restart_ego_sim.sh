#!/bin/bash
# Kill old simulator and restart
pkill -f "mujoco_ego_sim" 2>/dev/null
sleep 1
export PYTHONPATH=/workspace/umi
nohup python3 -u /workspace/umi/stage_2/simulation/mujoco_ego_sim.py --port 9999 \
    > /tmp/ego_sim.log 2>&1 &
echo "Simulator PID=$!"
sleep 3
cat /tmp/ego_sim.log
