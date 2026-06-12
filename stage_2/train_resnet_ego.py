#!/usr/bin/env python3
"""
EGO policy: ResNet18 image → action (NO state input).

Uses pretrained ResNet18 backbone + custom regression head.
Trains on color-cued data where only ONE marker is visible.
If this works, EGO is proven — the model extracts spatial info from images.

Training strategy:
  - Pretrained ResNet18 (ImageNet), fine-tuned
  - 50K steps with cosine LR schedule
  - Batch size 128 for stable gradients
  - Will test: same state, different marker → different action?
"""
import sys, os, time, glob
sys.path.insert(0, "/workspace/umi")
import numpy as np, h5py
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models


class ImageOnlyDataset(Dataset):
    def __init__(self, data_dir: str):
        self.samples = []
        for h5_path in sorted(glob.glob(os.path.join(data_dir, "*.h5"))):
            with h5py.File(h5_path, "r") as f:
                eps = [k for k in f.keys() if k.startswith("episode_")]
                if not eps: continue
                ep = f[eps[0]]
                if "sensors/camera/ego" not in ep: continue
                images = ep["sensors/camera/ego"][:]
                actions = ep["joint_command/position"][:]
                for i in range(len(images)):
                    self.samples.append((
                        images[i].astype(np.float32) / 255.0,
                        actions[i].astype(np.float32),
                    ))
        print(f"ResNet EGO Dataset: {len(self.samples)} frames")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, action = self.samples[idx]
        return (torch.from_numpy(img).permute(2, 0, 1),
                torch.from_numpy(action))


class ResNetEGOPolicy(nn.Module):
    """Pretrained ResNet18 → action regression."""
    def __init__(self, action_dim=6):
        super().__init__()
        # Pretrained ResNet18, remove classification head
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # → 512-dim

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, action_dim),
        )

    def forward(self, img):
        # ResNet expects 224×224; we have 64×64
        img_upscaled = F.interpolate(img, size=(224, 224), mode='bilinear',
                                      align_corners=False)
        feat = self.backbone(img_upscaled)  # (B, 512, 1, 1)
        feat = feat.flatten(1)              # (B, 512)
        return self.head(feat)


def main():
    device = torch.device("cuda")
    print(f"Device: {device}")

    ds = ImageOnlyDataset("/workspace/umi/data/color_cue_dataset")
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2,
                        pin_memory=True)

    model = ResNetEGOPolicy().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params (ResNet18 pretrained)")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50000)
    model.train()

    losses, best_loss, t0 = [], float("inf"), time.time()
    it = iter(loader)
    OUT = "/workspace/umi/outputs/ego_resnet"
    os.makedirs(OUT, exist_ok=True)

    N_STEPS = 50000
    for step in range(N_STEPS):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        losses.append(loss.item())

        if (step + 1) % 1000 == 0:
            a = np.mean(losses[-500:])
            print(f"  Step {step+1:5d}/{N_STEPS}: loss={loss.item():.4f} "
                  f"avg={a:.4f} {(step+1)/(time.time()-t0):.0f} s/s")

        if (step + 1) % 5000 == 0:
            al = np.mean(losses[-500:])
            if al < best_loss:
                best_loss = al
                torch.save({"step": step+1, "model_state_dict": model.state_dict()},
                           os.path.join(OUT, "best.pt"))
                print(f"    -> Best ({al:.4f})")

    elapsed = time.time() - t0
    print(f"\nDone: {elapsed:.0f}s, loss {losses[0]:.3f}→{np.mean(losses[-500:]):.4f}")
    torch.save({"step": N_STEPS, "model_state_dict": model.state_dict()},
               os.path.join(OUT, "final.pt"))
    print(f"Model: {OUT}/")


if __name__ == "__main__":
    main()
