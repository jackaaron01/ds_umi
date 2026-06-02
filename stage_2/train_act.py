#!/usr/bin/env python3
"""ACT training on UMI teleop data (mock or real)."""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature


class UMILeRobotDataset(Dataset):
    """Load LeRobot v3.0 parquet dataset for ACT training."""

    def __init__(self, data_dir: str, chunk_size: int = 100):
        self.chunk_size = chunk_size
        dfs = []
        data_path = os.path.join(data_dir, "data")
        for root, _, files in os.walk(data_path):
            for f in sorted(files):
                if f.endswith(".parquet"):
                    dfs.append(pd.read_parquet(os.path.join(root, f)))
        self.df = pd.concat(dfs, ignore_index=True)

        self.episodes = []
        self.samples = []
        for ep_idx, group in self.df.groupby("episode_index"):
            obs = np.vstack(group["observation.joint_position"].values).astype(np.float32)
            act = np.vstack(group["action.joint_position"].values).astype(np.float32)
            ep_len = len(act)
            list_idx = len(self.episodes)
            self.episodes.append((obs, act))
            for f in range(max(1, ep_len - chunk_size)):
                self.samples.append((list_idx, f))

        print(f"Dataset: {len(self.episodes)} eps, {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.samples[idx]
        obs_arr, act_arr = self.episodes[ep_idx]
        obs = torch.from_numpy(obs_arr[frame_idx])
        end = min(frame_idx + self.chunk_size, len(act_arr))
        action = np.zeros((self.chunk_size, 6), dtype=np.float32)
        n_valid = end - frame_idx
        if n_valid > 0:
            action[:n_valid] = act_arr[frame_idx:end]
        return obs, torch.from_numpy(action)


def main():
    parser = argparse.ArgumentParser(description="Train ACT on UMI data")
    parser.add_argument("--data", default="/workspace/umi/data/act_dataset_50")
    parser.add_argument("--output", default="/workspace/umi/outputs/act_baseline")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--dim-model", type=int, default=256)
    parser.add_argument("--save-every", type=int, default=2000)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    # Config
    cfg = ACTConfig(
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        n_obs_steps=1,
        input_features={"observation.environment_state": PolicyFeature(shape=[6], type=FeatureType.ENV)},
        output_features={"action": PolicyFeature(shape=[6], type=FeatureType.ACTION)},
        dim_model=args.dim_model,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=1,
        dim_feedforward=3200,
        dropout=0.1,
        use_vae=False,
    )

    # Dataset
    dataset = UMILeRobotDataset(args.data, chunk_size=args.chunk_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2 if device.type == "cuda" else 0,
                        pin_memory=(device.type == "cuda"))

    # Model
    model = ACTPolicy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, dim={args.dim_model}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)

    # Training
    model.train()
    losses = []
    best_loss = float("inf")
    t0 = time.time()
    data_iter = iter(loader)

    print(f"\nTraining {args.steps} steps, batch={args.batch_size}...")
    for step in range(args.steps):
        try:
            obs, action = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            obs, action = next(data_iter)

        obs = obs.to(device)
        action = action.to(device)

        optimizer.zero_grad()
        batch = {
            "observation.environment_state": obs,
            "observation.state": obs.unsqueeze(1),
            "action": action,
            "action_is_pad": torch.zeros(obs.size(0), args.chunk_size, dtype=torch.bool, device=device),
        }
        loss, info = model.forward(batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg = np.mean(losses[-500:])
            elapsed = time.time() - t0
            steps_per_sec = (step + 1) / elapsed
            print(f"  Step {step+1:5d}/{args.steps}: loss={loss.item():.4f}, "
                  f"avg500={avg:.4f}, lr={scheduler.get_last_lr()[0]:.2e}, "
                  f"{steps_per_sec:.0f} steps/s")

        if (step + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_{step+1:06d}.pt")
            torch.save({
                "step": step + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cfg": cfg,
                "losses": losses,
            }, ckpt_path)

            avg_loss = np.mean(losses[-500:])
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = os.path.join(args.output, "best.pt")
                torch.save({"step": step + 1, "model_state_dict": model.state_dict(), "cfg": cfg}, best_path)
                print(f"  → Best model saved ({avg_loss:.4f})")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({args.steps/elapsed:.0f} steps/s)")

    # Final stats
    first_500 = np.mean(losses[:500])
    last_500 = np.mean(losses[-500:])
    print(f"Loss: first500={first_500:.4f}, last500={last_500:.4f}, "
          f"ratio={last_500/max(first_500,1e-10):.3f}")

    # Save final model
    final_path = os.path.join(args.output, "final.pt")
    torch.save({"step": args.steps, "model_state_dict": model.state_dict(), "cfg": cfg}, final_path)
    print(f"Final model: {final_path}")
    print(f"Output dir: {args.output}")


if __name__ == "__main__":
    main()
