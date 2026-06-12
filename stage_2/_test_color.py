#!/usr/bin/env python3
"""Test: does the model move toward the visible marker based on image?"""
import sys, os, glob
sys.path.insert(0, "/workspace/umi")
os.environ["MUJOCO_GL"] = "glx"
import torch, numpy as np, mujoco, h5py
from stage_2.train_ego_cnn import EgoCNNPolicy
from stage_2.mujoco_renderer import MuJoCoRenderer

device = torch.device("cuda")

model = EgoCNNPolicy().to(device)
ckpt = torch.load("/workspace/umi/outputs/ego_cnn_color/best.pt",
                  map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

mpath = "/workspace/umi/stage_2/simulation/xarm6.xml"
m = mujoco.MjModel.from_xml_path(mpath)
renderer = MuJoCoRenderer(m, width=64, height=64)
rng = np.random.RandomState(42)
PHYS = 16

print("Color-cued EGO test:")
print("=" * 50)

# Test: red marker vs green marker at DIFFERENT positions
# Same start state, different marker positions → different actions?
for test_name, red_pos, green_pos in [
    ("Red LEFT, Green RIGHT",
     np.array([-0.2, 0.1, 0.35]), np.array([0.25, 0.1, 0.35])),
    ("Red LOW, Green HIGH",
     np.array([0.1, 0.1, 0.15]), np.array([0.1, 0.1, 0.45])),
]:
    actions = {}
    for label, target_pos, hide_pos in [
        ("red_target", red_pos, green_pos),
        ("green_target", green_pos, red_pos),
    ]:
        d = mujoco.MjData(m)
        d.qpos[:6] = np.array([0, -0.5, 0, 1.5, 0, 0])  # home position
        # Set mocap: target visible, other hidden
        if m.nmocap >= 2:
            if "red" in label:
                d.mocap_pos[0] = target_pos
                d.mocap_pos[1] = np.array([10, 10, -5])
            else:
                d.mocap_pos[0] = np.array([10, 10, -5])
                d.mocap_pos[1] = target_pos
        mujoco.mj_forward(m, d)

        # Get prediction from ego image
        ego_img = renderer.render(d, camera="ego")
        img_t = torch.from_numpy(ego_img.astype(np.float32) / 255.0
                                ).permute(2, 0, 1).unsqueeze(0).to(device)
        state_t = torch.from_numpy(d.qpos[:6].astype(np.float32)).unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(img_t, state_t)[0].cpu().numpy()
        actions[label] = action

    # Compare: do red-target and green-target produce different actions?
    diff = np.linalg.norm(actions["red_target"] - actions["green_target"])
    print(f"\n{test_name}:")
    print(f"  Red action:   {np.round(actions['red_target'], 2)}")
    print(f"  Green action: {np.round(actions['green_target'], 2)}")
    verdict = "DIFFERENT directions!" if diff > 0.1 else "Same direction (ignores image)"
    print(f"  |diff| = {diff:.4f} rad ({verdict})")

    # Which direction did each action point?
    red_dir = actions["red_target"] - np.array([0, -0.5, 0, 1.5, 0, 0])
    green_dir = actions["green_target"] - np.array([0, -0.5, 0, 1.5, 0, 0])
    angle = np.arccos(np.clip(np.dot(red_dir, green_dir) /
                              (np.linalg.norm(red_dir) * np.linalg.norm(green_dir) + 1e-8), -1, 1))
    dir_verdict = "Opposite" if np.degrees(angle) > 90 else "Similar"
    print(f"  Angle between directions: {np.degrees(angle):.0f} deg ({dir_verdict})")

renderer.close()
