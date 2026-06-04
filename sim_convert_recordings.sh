#!/bin/bash
# Convert recorded HDF5 episodes to LeRobot v3.0 and run quality check.
# Usage: bash sim_convert_recordings.sh [INPUT_DIR] [OUTPUT_DIR]

INPUT="${1:-/tmp/sim_teleop_recordings}"
OUTPUT="${2:-/workspace/umi/data/teleop_dataset}"
N_H5=$(ls "$INPUT"/*.h5 2>/dev/null | wc -l)

echo "Converting $N_H5 episodes from $INPUT"
echo "Output: $OUTPUT"

source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/workspace/umi:/ros2_ws/install/lib/python3.10/site-packages:$PYTHONPATH

# Convert to LeRobot v3.0
python3 -c "
from stage_2.lerobot_v3_converter import convert_directory, write_info_json, write_episodes_metadata, compute_and_write_stats
import pandas as pd, os, shutil
d_in = '$INPUT'
d_out = '${OUTPUT}_v3'
if os.path.exists(d_out):
    shutil.rmtree(d_out)
stats = convert_directory(d_in, d_out)
write_info_json(d_out, stats)
write_episodes_metadata(d_out, stats.episodes)
data_dir = os.path.join(d_out, 'data', 'chunk-000')
if os.path.isdir(data_dir):
    all_dfs = [pd.read_parquet(os.path.join(data_dir, f)) for f in sorted(os.listdir(data_dir)) if f.endswith('.parquet')]
    compute_and_write_stats(d_out, all_dfs)
print(f'Converted {stats.total_episodes} eps, {stats.total_frames} frames -> {d_out}')
"

echo ""
echo "Ready to train:"
echo "  python3 stage_2/train_act.py --data ${OUTPUT}_v3 --steps 10000"
