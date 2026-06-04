#!/bin/bash
# End-to-end pipeline: data generation -> ACT training -> DP training -> comparison
# Usage: bash run_full_pipeline.sh [DATA_DIR] [OUTPUT_DIR]
set -e

DATA="${1:-/workspace/umi/data/pipeline_dataset}"
OUTPUT="${2:-/workspace/umi/outputs/pipeline_run}"
N_EPISODES=300
SEED=42

echo "============================================"
echo "  UMI Full Training Pipeline"
echo "============================================"
echo "  Data:   $DATA"
echo "  Output: $OUTPUT"
echo "  Eps:    $N_EPISODES"
echo "============================================"
echo ""

# Step 1: Generate diverse training data
echo "[1/4] Generating $N_EPISODES episodes..."
python3 "$(dirname "$0")/generate_diverse_data.py" \
    -n "$N_EPISODES" -o "$DATA" --v3 --seed "$SEED"
echo "  Done: $DATA"
echo ""

# Step 2: Train ACT
echo "[2/4] Training ACT (20,000 steps)..."
mkdir -p "$OUTPUT/act"
python3 "$(dirname "$0")/train_act.py" \
    --data "${DATA}_v3" --output "$OUTPUT/act" \
    --steps 20000 --batch-size 32 --dim-model 256
echo "  Done: $OUTPUT/act"
echo ""

# Step 3: Train Diffusion Policy
echo "[3/4] Training Diffusion Policy (5,000 steps)..."
mkdir -p "$OUTPUT/dp"
python3 "$(dirname "$0")/train_dp.py" \
    --data "${DATA}_v3" --output "$OUTPUT/dp" \
    --steps 5000 --batch-size 32
echo "  Done: $OUTPUT/dp"
echo ""

# Step 4: Compare models
echo "[4/4] Comparing models..."
echo ""

# Direct comparison
python3 -c "
import sys; sys.path.insert(0, '/workspace/umi')
from stage_2.compare_models import load_checkpoint, measure_inference_speed, compute_test_mse
import torch, time, os, numpy as np, pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '${DATA}_v3'

print('=' * 70)
print(f'{\"Metric\":<25} {\"ACT\":>20} {\"Diffusion Policy\":>20}')
print('=' * 70)

# ACT
act_model, act_cfg, act_params, act_step, _ = load_checkpoint('$OUTPUT/act/best.pt', device)
act_speed = measure_inference_speed(act_model, act_cfg, device, model_type='act')
act_test = compute_test_mse(act_model, act_cfg, device, 'act', data_dir)

# DP
dp_model, dp_cfg, dp_params, dp_step, _ = load_checkpoint('$OUTPUT/dp/best.pt', device)
dp_speed = measure_inference_speed(dp_model, dp_cfg, device, model_type='dp')
dp_test = compute_test_mse(dp_model, dp_cfg, device, 'dp', data_dir)

rows = [
    ('Parameters', f'{act_params:,}', f'{dp_params:,}'),
    ('Training steps', str(act_step), str(dp_step)),
    ('Test MSE', f'{act_test:.6f}', f'{dp_test:.6f}'),
    ('Speed (samp/s)', f'{act_speed:.0f}', f'{dp_speed:.0f}'),
]
for name, a, d in rows:
    print(f'{name:<25} {a:>20} {d:>20}')
print('=' * 70)
print(f'  DP/ACT param ratio: {dp_params/act_params:.1f}x')
print(f'  DP/ACT test error:  {dp_test/max(act_test,1e-10):.2f}x')
print(f'  ACT/DP speed ratio: {act_speed/max(dp_speed,1e-10):.1f}x')
"

echo ""
echo "============================================"
echo "  Pipeline complete!"
echo "  Results: $OUTPUT"
echo "============================================"
