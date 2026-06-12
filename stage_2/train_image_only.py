#!/usr/bin/env python3
"""
IMAGE-ONLY policy: predict actions from ego images without joint state.

This FORCES the model to use image content. If it can predict actions
from images alone, EGO works. The state is completely removed.
"""
import sys, os, time, glob
sys.path.insert(0, "/workspace/umi")
import numpy as np, h5py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class ImageOnlyDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []
        for h5_path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
            with h5py.File(h5_path, "r") as f:
                ep = f[list(f.keys())[0]]
                if "sensors/camera/ego" not in ep:
                    continue
                images = ep["sensors/camera/ego"][:]
                actions = ep["joint_command/position"][:]
                for i in range(len(images)):
                    self.samples.append((
                        images[i].astype(np.float32) / 255.0,
                        actions[i].astype(np.float32),
                    ))
        print(f"ImageOnly Dataset: {len(self.samples)} frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, action = self.samples[idx]
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.from_numpy(action))


class ImageOnlyPolicy(nn.Module):
    """Predict actions from images only (no state)."""
    def __init__(self, action_dim=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, img):
        return self.head(self.conv(img))


def main():
    device = torch.device("cuda")
    print(f"Device: {device}")

    ds = ImageOnlyDataset("/workspace/umi/data/color_cue_dataset")
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2,
                        pin_memory=True)

    model = ImageOnlyPolicy().to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params (IMAGE ONLY)")

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10000)
    model.train()

    losses, best_loss, t0 = [], float("inf"), time.time()
    it = iter(loader)
    OUT = "/workspace/umi/outputs/ego_img_only"
    os.makedirs(OUT, exist_ok=True)

    for step in range(10000):
        try:
            img, action = next(it)
        except StopIteration:
            it = iter(loader)
            img, action = next(it)
        img, action = img.to(device), action.to(device)

        opt.zero_grad()
        pred = model(img)
        loss = F.mse_loss(pred, action)
        loss.backward()
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
                torch.save({"step": step+1, "model_state_dict": model.state_dict()},
                           os.path.join(OUT, "best.pt"))

    print(f"\nDone: {time.time()-t0:.0f}s, loss {losses[0]:.3f}→{np.mean(losses[-500:]):.4f}")
    torch.save({"step": 10000, "model_state_dict": model.state_dict()},
               os.path.join(OUT, "final.pt"))


if __name__ == "__main__":
    main()
