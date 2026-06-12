#!/usr/bin/env python3
"""
Train an image-aware EGO policy with a jointly-trained CNN encoder.

Architecture:
  Image (64×64×3) → Small ConvNet → 128-dim image features
  Joint state (6)  → FC(64) → 64-dim state features
  Concatenate(128+64=192) → FC(128) → FC(6) action output

Unlike frozen ResNet + ACT, this jointly trains the visual encoder
with the policy, allowing images to actually influence behavior.

The model learns to extract spatial information (marker position) from
ego-centric camera images, since the marker is ONLY visible in images.

Usage:
    python3 train_ego_cnn.py
"""
import sys, os, time, glob
sys.path.insert(0, "/workspace/umi")
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ═══════════════════════════════════════════════════════════════
class EgoCNNDataset(Dataset):
    """Load ego-centric images + joint states from HDF5 files."""

    def __init__(self, data_dir: str, device: str = "cuda"):
        self.device = device
        self.samples = []

        h5_files = sorted(glob.glob(os.path.join(data_dir, "*.h5")))
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                eps = [k for k in f.keys() if k.startswith("episode_")]
                if not eps:
                    continue
                ep = f[eps[0]]
                if "sensors/camera/ego" not in ep:
                    continue

                images = ep["sensors/camera/ego"][:]  # (N, 64, 64, 3)
                states = ep["joint_state/position"][:]  # (N, 6)
                actions = ep["joint_command/position"][:]  # (N, 6)

                n = len(images)
                for i in range(n):
                    self.samples.append((
                        images[i].astype(np.float32) / 255.0,  # normalize
                        states[i].astype(np.float32),
                        actions[i].astype(np.float32),
                    ))

        print(f"EgoCNN Dataset: {len(self.samples)} frames from {len(h5_files)} episodes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, state, action = self.samples[idx]
        return (
            torch.from_numpy(img).permute(2, 0, 1),  # (3, 64, 64)
            torch.from_numpy(state),
            torch.from_numpy(action),
        )


# ═══════════════════════════════════════════════════════════════
class EgoCNNPolicy(nn.Module):
    """Small CNN + MLP that jointly processes images and state."""

    def __init__(self, img_h=64, img_w=64, state_dim=6, action_dim=6):
        super().__init__()

        # Image encoder: deeper ConvNet with residual connections
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 32×32×32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 64×16×16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 128×8×8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),# 256×4×4
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),# 256×4×4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),  # 256
        )

        # State encoder
        self.state_fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self._img_dim = 256
        self._state_dim = 128

        # Combined head
        combined_dim = self._img_dim + self._state_dim  # 384
        self.head = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, img, state):
        img_feat = self.conv(img)            # (B, 256)
        state_feat = self.state_fc(state)     # (B, 128)
        combined = torch.cat([img_feat, state_feat], dim=-1)  # (B, 384)
        return self.head(combined)            # (B, 6)


# ═══════════════════════════════════════════════════════════════
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    ds = EgoCNNDataset("/workspace/umi/data/vary_marker_dataset")
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2,
                        pin_memory=(device.type == "cuda"))

    # Model
    model = EgoCNNPolicy().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters (CNN jointly trained)")

    # Training
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000)
    model.train()

    losses = []
    best_loss = float("inf")
    t0 = time.time()
    it = iter(loader)
    OUT = "/workspace/umi/outputs/ego_cnn_vary"
    os.makedirs(OUT, exist_ok=True)

    N_STEPS = 10000
    for step in range(N_STEPS):
        try:
            img, state, action = next(it)
        except StopIteration:
            it = iter(loader)
            img, state, action = next(it)

        img = img.to(device)
        state = state.to(device)
        action = action.to(device)

        opt.zero_grad()
        pred = model(img, state)
        loss = F.mse_loss(pred, action)
        loss.backward()
        opt.step()
        sch.step()
        losses.append(loss.item())

        if (step + 1) % 500 == 0:
            avg = np.mean(losses[-500:])
            print(f"  Step {step+1:5d}/{N_STEPS}: loss={loss.item():.4f} "
                  f"avg={avg:.4f} {(step+1)/(time.time()-t0):.0f} s/s")

        if (step + 1) % 2000 == 0:
            al = np.mean(losses[-500:])
            if al < best_loss:
                best_loss = al
                torch.save({"step": step + 1, "model_state_dict": model.state_dict()},
                           os.path.join(OUT, "best.pt"))
                print(f"    -> Best ({al:.4f})")

    elapsed = time.time() - t0
    print(f"\nDone: {elapsed:.0f}s, loss {losses[0]:.3f}→{np.mean(losses[-500:]):.4f}")
    torch.save({"step": N_STEPS, "model_state_dict": model.state_dict()},
               os.path.join(OUT, "final.pt"))
    print(f"Model: {OUT}/")


if __name__ == "__main__":
    main()
