#!/usr/bin/env python3
"""
Train a goal-conditioned Diffusion Policy.

Input: (joint_state, goal_position) → action
Compares with goal-conditioned ACT on the same data.

Usage:
    python3 train_goal_dp.py --data data/goal_la_dataset_v3 --steps 10000
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType, PolicyFeature


class GoalDPDataset(Dataset):
    """Load goal-conditioned data for Diffusion Policy."""

    def __init__(self, data_dir: str, n_obs_steps: int = 2, horizon: int = 16):
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        dfs = []
        data_path = os.path.join(data_dir, "data")
        for root, _, files in os.walk(data_path):
            for f in sorted(files):
                if f.endswith(".parquet"):
                    dfs.append(pd.read_parquet(os.path.join(root, f)))
        self.df = pd.concat(dfs, ignore_index=True)

        self._has_goal = "observation.goal_position" in self.df.columns
        self._has_image = "observation.image_features" in self.df.columns

        self.episodes = []
        self.samples = []
        for ep_idx, group in self.df.groupby("episode_index"):
            obs = np.vstack(group["observation.joint_position"].values).astype(np.float32)
            act = np.vstack(group["action.joint_position"].values).astype(np.float32)
            goal = None
            img_feat = None
            if self._has_goal:
                goal = np.vstack(group["observation.goal_position"].values).astype(np.float32)
            if self._has_image:
                img_feat = np.vstack(group["observation.image_features"].values).astype(np.float32)
            ep_len = len(act)
            list_idx = len(self.episodes)
            self.episodes.append((obs, act, goal, img_feat))
            for f in range(max(n_obs_steps, ep_len - horizon)):
                self.samples.append((list_idx, f))

        info = []
        if self._has_goal: info.append("goal")
        if self._has_image: info.append("image")
        print(f"Goal DP Dataset: {len(self.episodes)} eps, {len(self.samples)} samples "
              f"({' + '.join(info) if info else 'state-only'})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.samples[idx]
        obs_arr, act_arr, goal_arr, img_arr = self.episodes[ep_idx]

        # Observation history: (n_obs_steps, 6)
        obs_seq = np.zeros((self.n_obs_steps, 6), dtype=np.float32)
        for t in range(self.n_obs_steps):
            src = max(0, frame_idx - self.n_obs_steps + 1 + t)
            if src < len(obs_arr):
                obs_seq[t] = obs_arr[src]

        # Environment state: repeat current frame
        current_obs = obs_arr[min(frame_idx, len(obs_arr) - 1)]
        env_state = np.tile(current_obs, (self.n_obs_steps, 1))

        # Action horizon
        action = np.zeros((self.horizon, 6), dtype=np.float32)
        end = min(frame_idx + self.horizon, len(act_arr))
        n_valid = end - frame_idx
        if n_valid > 0:
            action[:n_valid] = act_arr[frame_idx:end]

        result = [
            torch.from_numpy(obs_seq),
            torch.from_numpy(env_state),
            torch.from_numpy(action),
        ]
        if self._has_goal and goal_arr is not None:
            # Goal: same for all timesteps
            goal_t = goal_arr[min(frame_idx, len(goal_arr) - 1)]
            goal_seq = np.tile(goal_t, (self.n_obs_steps, 1))
            result.insert(2, torch.from_numpy(goal_seq))
        if self._has_image and img_arr is not None:
            img_t = img_arr[min(frame_idx, len(img_arr) - 1)]
            img_seq = np.tile(img_t, (self.n_obs_steps, 1))
            result.append(torch.from_numpy(img_seq))

        return tuple(result)


def main():
    parser = argparse.ArgumentParser(description="Train goal-conditioned DP")
    parser.add_argument("--data", default="/workspace/umi/data/goal_la_dataset_v3")
    parser.add_argument("--output", default="/workspace/umi/outputs/dp_goal_la")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--no-visual", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = GoalDPDataset(args.data, n_obs_steps=2, horizon=16)
    has_goal = dataset._has_goal
    has_image = dataset._has_image and not args.no_visual
    if args.no_visual:
        dataset._has_image = False
    assert has_goal, "Need goal_position for goal-conditioned DP!"

    input_features = {
        "observation.environment_state": PolicyFeature(shape=[6], type=FeatureType.ENV),
        "observation.state": PolicyFeature(shape=[6], type=FeatureType.STATE),
        "observation.goal_position": PolicyFeature(shape=[6], type=FeatureType.ENV),
    }
    if has_image:
        input_features["observation.image_features"] = PolicyFeature(
            shape=[128], type=FeatureType.ENV
        )

    cfg = DiffusionConfig(
        n_obs_steps=2, horizon=16, n_action_steps=8,
        input_features=input_features,
        output_features={"action": PolicyFeature(shape=[6], type=FeatureType.ACTION)},
        down_dims=(256, 512, 1024),
        kernel_size=5, n_groups=8,
        diffusion_step_embed_dim=128,
        num_train_timesteps=100, num_inference_steps=10,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2 if device.type == "cuda" else 0,
                        pin_memory=(device.type == "cuda"))

    model = DiffusionPolicy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    model.train()
    losses = []
    best_loss = float("inf")
    t0 = time.time()
    data_iter = iter(loader)

    print(f"\nTraining {args.steps} steps, batch={args.batch_size}...")
    for step in range(args.steps):
        try: batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(loader); batch_data = next(data_iter)

        # Parse: obs_seq, env_state, [goal_seq], action, [img_seq]
        idx = 0
        obs_seq = batch_data[idx].to(device); idx += 1
        env_state = batch_data[idx].to(device); idx += 1
        goal_seq = batch_data[idx].to(device) if has_goal else None; idx += int(has_goal)
        action = batch_data[idx].to(device); idx += 1
        img_seq = batch_data[idx].to(device) if has_image else None

        optimizer.zero_grad()
        batch = {
            "observation.state": obs_seq,
            "observation.environment_state": env_state,
            "action": action,
            "action_is_pad": torch.zeros(action.size(0), cfg.horizon, dtype=torch.bool, device=device),
        }
        if has_goal:
            batch["observation.goal_position"] = goal_seq
        if has_image:
            batch["observation.image_features"] = img_seq

        loss, info = model.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg = np.mean(losses[-500:])
            elapsed = time.time() - t0
            sps = (step + 1) / elapsed
            print(f"  Step {step+1:5d}/{args.steps}: loss={loss.item():.4f}, "
                  f"avg500={avg:.4f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                  f"{sps:.0f} steps/s")

        if (step + 1) % args.save_every == 0:
            avg_loss = np.mean(losses[-500:])
            torch.save({"step": step + 1, "model_state_dict": model.state_dict(),
                         "cfg": cfg}, os.path.join(args.output, f"checkpoint_{step+1:06d}.pt"))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({"step": step + 1, "model_state_dict": model.state_dict(),
                             "cfg": cfg}, os.path.join(args.output, "best.pt"))
                print(f"  -> Best ({avg_loss:.4f})")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s ({args.steps/elapsed:.0f} steps/s)")
    print(f"Loss: first500={np.mean(losses[:500]):.4f}, last500={np.mean(losses[-500:]):.4f}")
    torch.save({"step": args.steps, "model_state_dict": model.state_dict(), "cfg": cfg},
               os.path.join(args.output, "final.pt"))


if __name__ == "__main__":
    main()
