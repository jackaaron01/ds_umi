#!/usr/bin/env python3
"""Train EGO policy: state + image_features → action (NO goal_position)."""
import sys, os, time, shutil
sys.path.insert(0, "/workspace/umi")
import torch, numpy as np, pandas as pd
from torch.utils.data import DataLoader
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from stage_2.lerobot_v3_converter import (
    convert_directory, write_info_json, write_episodes_metadata,
    compute_and_write_stats,
)
from stage_2.train_goal_act import GoalConditionedDataset

device = torch.device("cuda")
print(f"Device: {device}")

# Step 1: Convert to v3
v3dir = "/workspace/umi/data/ego_feat_dataset_v3"
if os.path.exists(v3dir):
    shutil.rmtree(v3dir)
stats = convert_directory("/workspace/umi/data/ego_feat_dataset", v3dir)
write_info_json(v3dir, stats)
write_episodes_metadata(v3dir, stats.episodes)
dd = os.path.join(v3dir, "data", "chunk-000")
dfs = [pd.read_parquet(os.path.join(dd, f))
       for f in sorted(os.listdir(dd)) if f.endswith(".parquet")]
compute_and_write_stats(v3dir, dfs)
print(f"v3: {v3dir}")

# Step 2: Load dataset WITHOUT goal_position
ds = GoalConditionedDataset(v3dir, chunk_size=100)
ds._has_goal = False  # FORCE image-only mode
print(f"EGO Dataset: {len(ds.episodes)} eps, goal=OFF, img=ON")

# Step 3: Train EGO policy
input_features = {
    "observation.environment_state": PolicyFeature(shape=[6], type=FeatureType.ENV),
    "observation.image_features": PolicyFeature(shape=[512], type=FeatureType.ENV),
}
cfg = ACTConfig(
    chunk_size=100, n_action_steps=100, n_obs_steps=1,
    input_features=input_features,
    output_features={"action": PolicyFeature(shape=[6], type=FeatureType.ACTION)},
    dim_model=256, n_heads=8, n_encoder_layers=4, n_decoder_layers=1,
    dim_feedforward=3200, dropout=0.1, use_vae=False)

loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
model = ACTPolicy(cfg).to(device)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000)
model.train()
losses, best_loss, t0 = [], float("inf"), time.time()
it = iter(loader)
outdir = "/workspace/umi/outputs/act_ego"
os.makedirs(outdir, exist_ok=True)

for step in range(10000):
    try:
        bd = next(it)
    except StopIteration:
        it = iter(loader)
        bd = next(it)
    obs, img, action = [bd[i].to(device) for i in range(3)]
    opt.zero_grad()
    batch = {
        "observation.environment_state": obs,
        "observation.image_features": img,
        "observation.state": obs.unsqueeze(1),
        "action": action,
        "action_is_pad": torch.zeros(obs.size(0), 100, dtype=torch.bool, device=device),
    }
    loss, _ = model.forward(batch)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sch.step()
    losses.append(loss.item())

    if (step + 1) % 500 == 0:
        a = np.mean(losses[-500:])
        print(f"  Step {step+1:5d}/10000: loss={loss.item():.4f} "
              f"avg={a:.4f} {(step+1)/(time.time()-t0):.0f} s/s")

    if (step + 1) % 2000 == 0:
        al = np.mean(losses[-500:])
        if al < best_loss:
            best_loss = al
            torch.save({"step": step + 1, "model_state_dict": model.state_dict(),
                         "cfg": cfg}, os.path.join(outdir, "best.pt"))
            print(f"    -> Best ({al:.4f})")

elapsed = time.time() - t0
print(f"\nDone: {elapsed:.0f}s, loss {losses[0]:.3f}→{np.mean(losses[-500:]):.4f}")
torch.save({"step": 10000, "model_state_dict": model.state_dict(), "cfg": cfg},
           os.path.join(outdir, "final.pt"))
print(f"Model: {outdir}/")
