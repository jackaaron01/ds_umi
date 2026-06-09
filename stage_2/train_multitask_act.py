#!/usr/bin/env python3
"""
Train a task-conditioned ACT policy on multi-task reaching data.

The model takes (joint_state, task_index) → action, learning to reach
different targets depending on the task index.

Usage:
    python3 train_multitask_act.py --data data/multitask_dataset_v3 --tasks 5
"""
import os, sys, time, argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType, PolicyFeature


class MultiTaskDataset(Dataset):
    """Load multi-task LeRobot v3.0 data for task-conditioned ACT."""

    def __init__(self, data_dir: str, chunk_size: int = 100, n_tasks: int = 5):
        self.chunk_size = chunk_size
        self.n_tasks = n_tasks
        dfs = []
        data_path = os.path.join(data_dir, "data")
        for root, _, files in os.walk(data_path):
            for f in sorted(files):
                if f.endswith(".parquet"):
                    dfs.append(pd.read_parquet(os.path.join(root, f)))
        self.df = pd.concat(dfs, ignore_index=True)

        self._has_image_features = "observation.image_features" in self.df.columns

        self.episodes = []
        self.samples = []
        for ep_idx, group in self.df.groupby("episode_index"):
            obs = np.vstack(group["observation.joint_position"].values).astype(np.float32)
            act = np.vstack(group["action.joint_position"].values).astype(np.float32)
            # Get task_index for this episode (should be constant within an episode)
            task_idx = int(group["task_index"].iloc[0]) if "task_index" in group else 0
            img_feat = None
            if self._has_image_features:
                img_feat = np.vstack(group["observation.image_features"].values).astype(np.float32)
            ep_len = len(act)
            list_idx = len(self.episodes)
            self.episodes.append((obs, act, img_feat, task_idx))
            for f in range(max(1, ep_len - chunk_size)):
                self.samples.append((list_idx, f))

        # Stats
        task_counts = {}
        for _, _, _, ti in self.episodes:
            task_counts[ti] = task_counts.get(ti, 0) + 1
        print(f"MultiTask Dataset: {len(self.episodes)} eps, {len(self.samples)} samples")
        print(f"  Tasks: {task_counts}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.samples[idx]
        obs_arr, act_arr, img_feat_arr, task_idx = self.episodes[ep_idx]

        obs = torch.from_numpy(obs_arr[frame_idx])
        task = torch.tensor(task_idx, dtype=torch.long)

        end = min(frame_idx + self.chunk_size, len(act_arr))
        action = np.zeros((self.chunk_size, 6), dtype=np.float32)
        n_valid = end - frame_idx
        if n_valid > 0:
            action[:n_valid] = act_arr[frame_idx:end]

        if self._has_image_features and img_feat_arr is not None:
            img_feat = torch.from_numpy(img_feat_arr[frame_idx])
            return obs, task, img_feat, torch.from_numpy(action)
        return obs, task, torch.from_numpy(action)


def main():
    parser = argparse.ArgumentParser(description="Train task-conditioned ACT")
    parser.add_argument("--data", default="/workspace/umi/data/multitask_dataset_v3")
    parser.add_argument("--output", default="/workspace/umi/outputs/act_multitask")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--dim-model", type=int, default=256)
    parser.add_argument("--tasks", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--no-visual", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Dataset
    dataset = MultiTaskDataset(args.data, chunk_size=args.chunk_size,
                               n_tasks=args.tasks)
    if args.no_visual:
        dataset._has_image_features = False
    has_visual = dataset._has_image_features

    # Config: include task_index as input
    input_features = {
        "observation.environment_state": PolicyFeature(shape=[6], type=FeatureType.ENV),
        "observation.task_index": PolicyFeature(shape=[args.tasks], type=FeatureType.ENV),
    }
    if has_visual:
        input_features["observation.image_features"] = PolicyFeature(
            shape=[128], type=FeatureType.ENV
        )

    cfg = ACTConfig(
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        n_obs_steps=1,
        input_features=input_features,
        output_features={"action": PolicyFeature(shape=[6], type=FeatureType.ACTION)},
        dim_model=args.dim_model,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=1,
        dim_feedforward=3200,
        dropout=0.1,
        use_vae=False,
    )

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2 if device.type == "cuda" else 0,
                        pin_memory=(device.type == "cuda"))

    model = ACTPolicy(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters, tasks={args.tasks}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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

        if has_visual:
            obs, task, img_feat, action = batch_data
            img_feat = img_feat.to(device)
        else:
            obs, task, action = batch_data

        obs = obs.to(device)
        task = task.to(device)
        action = action.to(device)

        # One-hot encode task index
        task_onehot = torch.zeros(task.size(0), args.tasks, device=device)
        task_onehot.scatter_(1, task.unsqueeze(1), 1.0)

        optimizer.zero_grad()
        batch = {
            "observation.environment_state": obs,
            "observation.task_index": task_onehot,
            "observation.state": obs.unsqueeze(1),
            "action": action,
            "action_is_pad": torch.zeros(obs.size(0), args.chunk_size, dtype=torch.bool, device=device),
        }
        if has_visual:
            batch["observation.image_features"] = img_feat

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
            ckpt_path = os.path.join(args.output, f"checkpoint_{step+1:06d}.pt")
            torch.save({"step": step + 1, "model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "cfg": cfg, "losses": losses}, ckpt_path)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({"step": step + 1, "model_state_dict": model.state_dict(),
                             "cfg": cfg, "n_tasks": args.tasks},
                           os.path.join(args.output, "best.pt"))
                print(f"  -> Best model saved ({avg_loss:.4f})")

    elapsed = time.time() - t0
    print(f"\nTraining done in {elapsed:.0f}s ({args.steps/elapsed:.0f} steps/s)")
    first_500 = np.mean(losses[:500])
    last_500 = np.mean(losses[-500:])
    print(f"Loss: first500={first_500:.4f}, last500={last_500:.4f}, "
          f"ratio={last_500/max(first_500,1e-10):.3f}")

    final_path = os.path.join(args.output, "final.pt")
    torch.save({"step": args.steps, "model_state_dict": model.state_dict(),
                 "cfg": cfg, "n_tasks": args.tasks}, final_path)
    print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
