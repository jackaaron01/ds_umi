#!/usr/bin/env python3
"""
Generate all result visualizations using Pillow (no matplotlib needed).

Generates:
1. Robot arm renders (MuJoCo offscreen)
2. Training loss curves (Pillow chart)
3. Model comparison charts (Pillow chart)
4. Pipeline diagram (Pillow)

Output: outputs/figures/
"""
import sys, os, json, glob
sys.path.insert(0, "/workspace/umi")
os.environ["MUJOCO_GL"] = "glx"

import mujoco, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import textwrap

OUTDIR = "/workspace/umi/outputs/figures"
os.makedirs(OUTDIR, exist_ok=True)

FONT = None
for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
           "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
    if os.path.exists(fp):
        FONT = ImageFont.truetype(fp, 14)
        FONT_SM = ImageFont.truetype(fp, 11)
        FONT_BIG = ImageFont.truetype(fp, 18)
        break
if FONT is None:
    FONT = ImageFont.load_default()

# ═══════════════════════════════════════════════════════════════
# 1. Robot arm renders
# ═══════════════════════════════════════════════════════════════
print("1. Robot arm renders...")

from stage_2.mujoco_renderer import MuJoCoRenderer

model_path = "/workspace/umi/stage_2/simulation/xarm6.xml"
model = mujoco.MjModel.from_xml_path(model_path)
renderer = MuJoCoRenderer(model, width=320, height=240)
data = mujoco.MjData(model)

poses = [
    ("home\n[0, -0.5, 0, 1.5, 0, 0]",
     [0.0, -0.5, 0.0, 1.5, 0.0, 0.0]),
    ("forward\n[0.5, -0.3, 0.3, 1.2, 0, 0]",
     [0.5, -0.3, 0.3, 1.2, 0.0, 0.0]),
    ("left\n[-0.6, -0.6, -0.4, 1.8, 0.4, 0]",
     [-0.6, -0.6, -0.4, 1.8, 0.4, 0.0]),
    ("high right\n[0.8, -0.2, 0.5, 0.8, -0.2, 0.5]",
     [0.8, -0.2, 0.5, 0.8, -0.2, 0.5]),
    ("extended\n[0, -0.8, 0.2, 2.1, 0, 0.5]",
     [0.0, -0.8, 0.2, 2.1, 0.0, 0.5]),
    ("compact\n[0, -0.2, -0.5, 1.0, -0.5, 0.2]",
     [0.0, -0.2, -0.5, 1.0, -0.5, 0.2]),
]

images = []
labels = []
for name, jpos in poses:
    data.qpos[:6] = jpos
    mujoco.mj_forward(model, data)
    img = renderer.render(data)
    images.append(Image.fromarray(img))
    labels.append(name)

renderer.close()

# Create grid: 2 rows × 3 cols
grid_w, grid_h = 320, 240
label_h = 35
gap = 6
cols, rows = 3, 2
canvas_w = cols * grid_w + (cols + 1) * gap
canvas_h = rows * (grid_h + label_h) + (rows + 1) * gap + 30
canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 35))
draw = ImageDraw.Draw(canvas)

# Title
draw.text((canvas_w // 2 - 100, 5), "xArm6 Robot Poses", fill=(255, 255, 255),
          font=FONT_BIG if FONT_BIG else FONT)

for idx, (img, label) in enumerate(zip(images, labels)):
    col = idx % cols
    row = idx // cols
    x = gap + col * (grid_w + gap)
    y = gap + 25 + row * (grid_h + label_h + gap)
    canvas.paste(img, (x, y))
    for line_idx, line in enumerate(label.split('\n')):
        draw.text((x + 5, y + grid_h + 2 + line_idx * 14), line,
                  fill=(200, 200, 210), font=FONT_SM if FONT_SM else FONT)

canvas.save(os.path.join(OUTDIR, "robot_poses.png"))
print(f"  Saved robot_poses.png ({canvas_w}x{canvas_h})")

# ═══════════════════════════════════════════════════════════════
# 2. Training loss curves (Pillow-drawn chart)
# ═══════════════════════════════════════════════════════════════
print("2. Training loss curves...")

# Gather loss data from checkpoints
loss_data = {}
for ckpt_dir in ["outputs/act_goal_la_big", "outputs/act_goal_la",
                  "outputs/act_goal", "outputs/act_state_only"]:
    ckpt_path = os.path.join("/workspace/umi", ckpt_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        continue
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "losses" in ckpt:
            name = ckpt_dir.replace("outputs/act_", "")
            loss_data[name] = ckpt["losses"]
            print(f"  {name}: {len(loss_data[name])} points, final={loss_data[name][-1]:.4f}")
    except Exception as e:
        print(f"  {ckpt_dir}: {e}")

if loss_data:
    # Chart dimensions
    cw, ch = 800, 400
    margin = 60
    chart = Image.new("RGB", (cw, ch), (25, 25, 30))
    draw = ImageDraw.Draw(chart)

    # Find global range
    all_losses = []
    for losses in loss_data.values():
        all_losses.extend(losses)
    y_min, y_max = 0.01, max(all_losses)

    # Y axis (log scale)
    for y_val in [0.01, 0.03, 0.1, 0.3, 1.0]:
        if y_val >= y_min and y_val <= y_max:
            y_px = margin + (ch - 2 * margin) * (1 - (np.log10(y_val) - np.log10(y_min)) /
                                                   (np.log10(y_max) - np.log10(y_min)))
            draw.line([(margin, y_px), (cw - margin, y_px)], fill=(60, 60, 65), width=1)
            draw.text((5, y_px - 7), f"{y_val:.2f}", fill=(180, 180, 185), font=FONT_SM)

    # Color palette
    colors = [(0, 200, 100), (0, 150, 255), (255, 150, 50), (255, 80, 80),
              (200, 100, 255), (255, 200, 50)]

    for idx, (name, losses) in enumerate(sorted(loss_data.items(),
                                                 key=lambda x: x[1][-1])):
        color = colors[idx % len(colors)]
        steps = np.arange(len(losses))

        # Smooth with moving average
        window = max(1, len(losses) // 300)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            s_steps = steps[window-1:]
        else:
            smoothed = losses
            s_steps = steps

        # Plot
        points = []
        for i in range(0, len(s_steps), max(1, len(s_steps)//500)):
            t = s_steps[i]
            log_val = np.log10(max(smoothed[i], y_min))
            x = margin + (t / max(steps)) * (cw - 2 * margin)
            y = margin + (ch - 2 * margin) * (1 - (log_val - np.log10(y_min)) /
                                               (np.log10(y_max) - np.log10(y_min)))
            points.append((x, y))

        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=color, width=2)

        # Label
        label = f"{name} ({losses[-1]:.3f})"
        draw.text((cw - margin - 200, margin + 5 + idx * 18), label,
                  fill=color, font=FONT_SM)

    # Axes
    draw.line([(margin, margin), (margin, ch - margin)], fill=(150, 150, 155), width=2)
    draw.line([(margin, ch - margin), (cw - margin, ch - margin)],
              fill=(150, 150, 155), width=2)
    draw.text((cw // 2 - 60, ch - 25), "Training Step", fill=(200, 200, 205), font=FONT)
    draw.text((15, ch // 2 - 30), "Loss\n(log)", fill=(200, 200, 205), font=FONT_SM)

    # Title
    draw.text((cw // 2 - 100, 10), "ACT Training Loss Curves (smoothed)",
              fill=(255, 255, 255), font=FONT_BIG)

    chart.save(os.path.join(OUTDIR, "training_loss_curves.png"))
    print(f"  Saved training_loss_curves.png ({cw}x{ch})")

# ═══════════════════════════════════════════════════════════════
# 3. Model comparison bar chart
# ═══════════════════════════════════════════════════════════════
print("3. Model comparison charts...")

eval_data = [
    ("state_only", 4.4, (100, 150, 255)),
    ("diverse_20k", 4.5, (100, 150, 255)),
    ("goal (K=0)", 12.7, (255, 180, 60)),
    ("goal_la (K=8-15)", 26.4, (255, 130, 40)),
    ("goal_la_big\n(K=20-30)", 38.6, (255, 70, 30)),
]

# Bar chart
cw, ch = 600, 300
margin_x, margin_y = 140, 50
chart = Image.new("RGB", (cw, ch), (25, 25, 30))
draw = ImageDraw.Draw(chart)

bar_w = 50
bar_gap = 30
max_val = 45
baseline_x = margin_x

for idx, (name, val, color) in enumerate(eval_data):
    x = baseline_x + idx * (bar_w + bar_gap)
    bar_h = int((val / max_val) * (ch - 2 * margin_y))
    y = ch - margin_y - bar_h

    # Bar
    draw.rectangle([x, y, x + bar_w, ch - margin_y], fill=color)

    # Value label
    draw.text((x + 5, y - 18), f"{val}%", fill=(255, 255, 255), font=FONT)

    # Name label (below bar, rotated or wrapped)
    for i, line in enumerate(name.split('\n')):
        draw.text((x + 2, ch - margin_y + 2 + i * 12), line, fill=(200, 200, 205), font=FONT_SM)

# Baseline
draw.line([(margin_x - 10, ch - margin_y), (cw - margin_x//2, ch - margin_y)],
          fill=(150, 150, 155), width=1)

# Axes
draw.line([(margin_x - 5, margin_y), (margin_x - 5, ch - margin_y)],
          fill=(150, 150, 155), width=2)

# Y ticks
for val in [0, 10, 20, 30, 40]:
    y = ch - margin_y - int((val / max_val) * (ch - 2 * margin_y))
    draw.line([(margin_x - 10, y), (margin_x - 2, y)], fill=(150, 150, 155), width=1)
    draw.text((margin_x - 35, y - 7), f"{val}%", fill=(180, 180, 185), font=FONT_SM)

# Title
draw.text((cw // 2 - 120, 8), "Goal-Reaching Improvement (higher = better)",
          fill=(255, 255, 255), font=FONT_BIG)

chart.save(os.path.join(OUTDIR, "model_comparison.png"))
print(f"  Saved model_comparison.png ({cw}x{ch})")

# Second chart: Generalization gap
cw2, ch2 = 650, 280
chart2 = Image.new("RGB", (cw2, ch2), (25, 25, 30))
draw2 = ImageDraw.Draw(chart2)

gen_data = [
    ("state_only", 4.6, 4.5),
    ("goal\n(K=0)", 13.4, 11.3),
    ("goal_la\n(K=8-15)", 24.9, 29.3),
    ("goal_la_big\n(K=20-30)", 38.4, 39.1),
]

mx, my = 130, 40
bar_w2 = 35
group_gap = 50
max_val2 = 45

for idx, (name, seen, unseen) in enumerate(gen_data):
    x = mx + idx * (2 * bar_w2 + group_gap)

    # Seen bar
    sh = int((seen / max_val2) * (ch2 - 2 * my))
    draw2.rectangle([x, ch2 - my - sh, x + bar_w2, ch2 - my],
                    fill=(52, 152, 219))
    draw2.text((x + 3, ch2 - my - sh - 18), f"{seen}%", fill=(255, 255, 255), font=FONT_SM)

    # Unseen bar
    uh = int((unseen / max_val2) * (ch2 - 2 * my))
    draw2.rectangle([x + bar_w2 + 5, ch2 - my - uh, x + 2 * bar_w2 + 5, ch2 - my],
                    fill=(231, 76, 60))
    draw2.text((x + bar_w2 + 8, ch2 - my - uh - 18), f"{unseen}%", fill=(255, 255, 255), font=FONT_SM)

    # Name
    for i, line in enumerate(name.split('\n')):
        draw2.text((x + 5, ch2 - my + 3 + i * 12), line, fill=(200, 200, 205), font=FONT_SM)

# Legend
draw2.rectangle([mx + 300, 20, mx + 315, 35], fill=(52, 152, 219))
draw2.text((mx + 320, 22), "Seen Goals", fill=(200, 200, 205), font=FONT_SM)
draw2.rectangle([mx + 400, 20, mx + 415, 35], fill=(231, 76, 60))
draw2.text((mx + 420, 22), "Unseen Goals", fill=(200, 200, 205), font=FONT_SM)

# Axes
draw2.line([(mx - 5, my), (mx - 5, ch2 - my)], fill=(150, 150, 155), width=2)
draw2.line([(mx - 5, ch2 - my), (cw2 - 50, ch2 - my)], fill=(150, 150, 155), width=2)
for val in [0, 10, 20, 30, 40]:
    y = ch2 - my - int((val / max_val2) * (ch2 - 2 * my))
    draw2.line([(mx - 10, y), (mx - 2, y)], fill=(150, 150, 155), width=1)
    draw2.text((mx - 35, y - 7), f"{val}%", fill=(180, 180, 185), font=FONT_SM)

draw2.text((cw2 // 2 - 140, 5), "Generalization: Seen vs Unseen Goals",
           fill=(255, 255, 255), font=FONT_BIG)

chart2.save(os.path.join(OUTDIR, "generalization_gap.png"))
print(f"  Saved generalization_gap.png ({cw2}x{ch2})")

# ═══════════════════════════════════════════════════════════════
# 4. Lookahead effect chart
# ═══════════════════════════════════════════════════════════════
print("4. Lookahead scaling chart...")

la_data = [("K=0\n(original)", 12.7, 0.08), ("K=8-15", 26.4, 0.39), ("K=20-30", 38.6, 0.54)]

cw3, ch3 = 500, 300
chart3 = Image.new("RGB", (cw3, ch3), (25, 25, 30))
draw3 = ImageDraw.Draw(chart3)
mx3, my3 = 80, 45

# Improvement bars
bar_w3 = 80
gap3 = 60
for idx, (name, improvement, obs_act) in enumerate(la_data):
    x = mx3 + idx * (bar_w3 + gap3)
    h = int((improvement / 45) * (ch3 - 2 * my3))
    y = ch3 - my3 - h
    color = [(100, 200, 100), (255, 150, 50), (255, 60, 30)][idx]
    draw3.rectangle([x, y, x + bar_w3, ch3 - my3], fill=color)
    draw3.text((x + 10, y - 30), f"{improvement}%", fill=(255, 255, 255), font=FONT_BIG)
    draw3.text((x + 8, y - 15), f"obs-act\ndiff={obs_act}", fill=(200, 200, 205), font=FONT_SM)
    for i, line in enumerate(name.split('\n')):
        draw3.text((x + 10, ch3 - my3 + 3 + i * 13), line, fill=(200, 200, 205), font=FONT_SM)

draw3.line([(mx3 - 5, my3), (mx3 - 5, ch3 - my3)], fill=(150, 150, 155), width=2)
draw3.line([(mx3 - 5, ch3 - my3), (cw3 - 30, ch3 - my3)], fill=(150, 150, 155), width=2)
for val in [0, 10, 20, 30, 40]:
    y = ch3 - my3 - int((val / 45) * (ch3 - 2 * my3))
    draw3.line([(mx3 - 10, y), (mx3 - 2, y)], fill=(150, 150, 155), width=1)
    draw3.text((mx3 - 40, y - 7), f"{val}%", fill=(180, 180, 185), font=FONT_SM)

draw3.text((cw3 // 2 - 120, 8), "Effect of Lookahead (K) on Goal Reaching",
           fill=(255, 255, 255), font=FONT_BIG)

chart3.save(os.path.join(OUTDIR, "lookahead_scaling.png"))
print(f"  Saved lookahead_scaling.png ({cw3}x{ch3})")

# ═══════════════════════════════════════════════════════════════
print(f"\nDone! All figures in {OUTDIR}/")
for f in sorted(os.listdir(OUTDIR)):
    size = os.path.getsize(os.path.join(OUTDIR, f))
    print(f"  {OUTDIR}/{f} ({size/1024:.0f} KB)")
