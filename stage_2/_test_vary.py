#!/usr/bin/env python3
"""Test: does varying-marker training enable image-based spatial reasoning?"""
import sys, os, glob
sys.path.insert(0, "/workspace/umi")
import torch, numpy as np, h5py
from stage_2.train_ego_cnn import EgoCNNPolicy

device = torch.device("cuda")

# Load model
model = EgoCNNPolicy().to(device)
ckpt = torch.load("/workspace/umi/outputs/ego_cnn_vary/best.pt",
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Load varying-marker data — find frames where marker changes
h5_files = sorted(glob.glob("/workspace/umi/data/vary_marker_dataset/episode_0000*.h5"))

# Find consecutive frames where marker position DIFFERS but state is SIMILAR
pairs = []
for h5_path in h5_files[:5]:
    with h5py.File(h5_path, "r") as f:
        ep = f[list(f.keys())[0]]
        markers = ep["observation/marker_position"][:]
        states = ep["joint_state/position"][:]
        images = ep["sensors/camera/ego"][:]

        for i in range(len(markers) - 5):
            # Find frames where marker moved significantly
            marker_diff = np.linalg.norm(markers[i+5] - markers[i])
            state_diff = np.linalg.norm(states[i+5] - states[i])
            if marker_diff > 0.1 and state_diff < 0.3:
                pairs.append((
                    images[i].astype(np.float32) / 255.0,
                    images[i+5].astype(np.float32) / 255.0,
                    states[i].astype(np.float32),
                    markers[i],
                    markers[i+5],
                ))
                if len(pairs) >= 10:
                    break
        if len(pairs) >= 10:
            break

print(f"Found {len(pairs)} test pairs (similar state, different marker)")
print()

# Test: same state, different images → different actions?
print("Same state, different marker images → different actions?")
max_diff = 0
for img1, img2, state, m1, m2 in pairs:
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).to(device)
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).to(device)
    state_t = torch.from_numpy(state).unsqueeze(0).to(device)

    with torch.no_grad():
        a1 = model(img1_t, state_t)[0].cpu().numpy()
        a2 = model(img2_t, state_t)[0].cpu().numpy()

    diff = np.linalg.norm(a1 - a2)
    max_diff = max(max_diff, diff)
    print(f"  Marker: {m1}→{m2} |action_diff|={diff:.4f} rad")

print(f"\nMax action diff: {max_diff:.4f} rad "
      f"({'USES image content!' if max_diff > 0.1 else 'Still ignores image content'}")

# Test with zero image vs real image
print("\nZero image vs real image — does model respond to image presence?")
state = torch.randn(1, 6, device=device) * 0.5
img_real = torch.from_numpy(pairs[0][0]).permute(2, 0, 1).unsqueeze(0).to(device)
img_zero = torch.zeros(1, 3, 64, 64, device=device)

with torch.no_grad():
    a_real = model(img_real, state)[0].cpu().numpy()
    a_zero = model(img_zero, state)[0].cpu().numpy()

print(f"  |action(zero_img) - action(real_img)| = {np.linalg.norm(a_zero - a_real):.4f} rad "
      f"({'Images matter' if np.linalg.norm(a_zero - a_real) > 0.1 else 'Images ignored'})")
