#!/usr/bin/env python3
"""Diffusion Policy training on UMI teleop data."""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.configs.types import FeatureType, PolicyFeature


class UMIDPDataset(Dataset):
    """Load LeRobot v3.0 data for Diffusion Policy (needs n_obs_steps=2)."""

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

        # Check if image_features column exists
        self._has_image_features = "observation.image_features" in self.df.columns

        self.episodes = []
        self.samples = []
        for ep_idx, group in self.df.groupby("episode_index"):
            obs = np.vstack(group["observation.joint_position"].values).astype(np.float32)
            act = np.vstack(group["action.joint_position"].values).astype(np.float32)
            img_feat = None
            if self._has_image_features:
                img_feat = np.vstack(group["observation.image_features"].values).astype(np.float32)
            ep_len = len(act)
            list_idx = len(self.episodes)
            self.episodes.append((obs, act, img_feat))
            for f in range(max(n_obs_steps, ep_len - horizon)):
                self.samples.append((list_idx, f))

        feat_str = " w/ image_features" if self._has_image_features else ""
        print(f"DP Dataset: {len(self.episodes)} eps, {len(self.samples)} samples{feat_str}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.samples[idx]
        obs_arr, act_arr, img_feat_arr = self.episodes[ep_idx]

        # Observation history: (n_obs_steps, 6)
        obs_seq = np.zeros((self.n_obs_steps, 6), dtype=np.float32)
        for t in range(self.n_obs_steps):
            src = max(0, frame_idx - self.n_obs_steps + 1 + t)
            if src < len(obs_arr):
                obs_seq[t] = obs_arr[src]

        # Environment state: repeat current frame for n_obs_steps
        current_obs = obs_arr[min(frame_idx, len(obs_arr) - 1)]
        env_state = np.tile(current_obs, (self.n_obs_steps, 1))

        # Action horizon: (horizon, 6)
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
        if self._has_image_features and img_feat_arr is not None:
            # Image features: (n_obs_steps, 128)
            img_feat_seq = np.zeros((self.n_obs_steps, img_feat_arr.shape[1]), dtype=np.float32)
            for t in range(self.n_obs_steps):
                src = max(0, frame_idx - self.n_obs_steps + 1 + t)
                if src < len(img_feat_arr):
                    img_feat_seq[t] = img_feat_arr[src]
            result.append(torch.from_numpy(img_feat_seq))

        return tuple(result)


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy on UMI data")
    parser.add_argument("--data", default="/workspace/umi/data/diverse_dataset_v3")
    parser.add_argument("--output", default="/workspace/umi/outputs/dp_diverse")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-every", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset (load first to detect image features)
    dataset = UMIDPDataset(args.data, n_obs_steps=2, horizon=16)
    has_image_features = dataset._has_image_features

    input_features = {
        "observation.environment_state": PolicyFeature(shape=[6], type=FeatureType.ENV),
        "observation.state": PolicyFeature(shape=[6], type=FeatureType.STATE),
    }
    if has_image_features:
        input_features["observation.image_features"] = PolicyFeature(
            shape=[128], type=FeatureType.ENV
        )

    cfg = DiffusionConfig(
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
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
        try:
            batch_data = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch_data = next(data_iter)

        if has_image_features:
            obs_seq, env_state, action, img_feat_seq = batch_data
            img_feat_seq = img_feat_seq.to(device)
        else:
            obs_seq, env_state, action = batch_data

        obs_seq = obs_seq.to(device)
        env_state = env_state.to(device)
        action = action.to(device)

        optimizer.zero_grad()
        batch = {
            "observation.state": obs_seq,
            "observation.environment_state": env_state,
            "action": action,
            "action_is_pad": torch.zeros(action.size(0), cfg.horizon, dtype=torch.bool, device=device),
        }
        if has_image_features:
            batch["observation.image_features"] = img_feat_seq
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
            ckpt_path = os.path.join(args.output, f"checkpoint_{step+1:06d}.pt")
            torch.save({"step": step+1, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "cfg": cfg, "losses": losses}, ckpt_path)
            avg_loss = np.mean(losses[-500:])
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output, "best.pt")
                torch.save({"step": step+1, "model_state_dict": model.state_dict(),
                             "cfg": cfg}, best_path)
                print(f"  -> Best model saved ({avg_loss:.4f})")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({args.steps/elapsed:.0f} steps/s)")
    first_500 = np.mean(losses[:500])
    last_500 = np.mean(losses[-500:])
    print(f"Loss: first500={first_500:.4f}, last500={last_500:.4f}, ratio={last_500/max(first_500,1e-10):.3f}")

    final_path = os.path.join(args.output, "final.pt")
    torch.save({"step": args.steps, "model_state_dict": model.state_dict(), "cfg": cfg}, final_path)
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
