# ACT 训练方案

> 基于 LeRobot v3.0 数据格式和 UMI 遥操作管道的 ACT（Action Chunking Transformer）训练方案。

## 1. 数据格式要点

### LeRobot v3.0 输出结构

```
dataset_v3/
├── data/chunk-000/file-000.parquet  ← 帧数据（episode_index, frame_index, index, timestamp, task_index + features）
├── meta/info.json                   ← 数据集配置（features 定义、fps、robot_type）
├── meta/stats.json                  ← 归一化统计量（min/max/mean/std per feature）
├── meta/tasks.parquet               ← 任务定义
└── meta/episodes/chunk-000/         ← episode 元数据（index, length）
```

### Feature 命名（点分隔）

| Feature Key | Shape | 说明 |
|-------------|-------|------|
| `action.joint_position` | (6,) | 6 关节目标角（rad） |
| `action.gripper` | (1,) | 夹爪目标开度 [0,1] |
| `observation.joint_position` | (6,) | 6 关节实际角（rad） |
| `observation.joint_velocity` | (6,) | 6 关节实际速度（rad/s） |
| `observation.gripper` | (1,) | 夹爪实际开度 |

当前为 **state-only** 模式（无相机图像）。如需视觉输入，添加 `observation.images.camera_rgb` 等 feature。

## 2. 从录制到训练的完整流程

### Step 1: 录制数据

```bash
# 在容器内启动遥操作管道（mock 或真实硬件）
ros2 launch launch teleop_mock.launch.py output_dir:=~/umi_recordings

# 通过 service 控制录制
ros2 service call /recorder/start std_srvs/srv/Trigger
# ... 操作演示 ...
ros2 service call /recorder/stop std_srvs/srv/Trigger
```

### Step 2: 转换为 LeRobot v3.0

```bash
# 批量转换（命令行）
python3 stage_2/lerobot_v3_converter.py \
    --input ~/umi_recordings \
    --output ./dataset_v3 \
    --fps 30 \
    --tasks stage_2/tasks/example_tasks.yaml

# 或通过 Python API（批量 mock 录制 + 转换）
python3 stage_2/act_research.py 5.0  # 录制 5 秒并转换为 v3
```

### Step 3: 验证数据集

```bash
# 检查 info.json
python3 -c "
import json
with open('dataset_v3/meta/info.json') as f:
    info = json.load(f)
print('Features:', list(info['features'].keys()))
print('Episodes:', info['total_episodes'])
print('Frames:', info['total_frames'])
"

# 检查 stats.json
python3 -c "
import json
with open('dataset_v3/meta/stats.json') as f:
    stats = json.load(f)
for k, v in stats.items():
    print(f'{k}: mean={v[\"mean\"]}, std={v[\"std\"]}')
"
```

### Step 4: ACT 训练

```bash
# LeRobot 命令行训练（需 GPU）
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=./dataset_v3 \
    --training.offline_steps=50000 \
    --training.batch_size=32 \
    --training.learning_rate=1e-4 \
    --policy.chunk_size=100 \
    --policy.n_action_steps=100 \
    --output_dir=./outputs/act_baseline

# 或使用 Python API
python3 -c "
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
# ... 训练代码见 LeRobot 官方示例
"
```

## 3. 建议超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `chunk_size` | 100 | 动作分块大小（预测未来 100 步 ≈ 3.3s @ 30Hz） |
| `n_action_steps` | 100 | 执行的动作步数（与 chunk_size 对齐） |
| `batch_size` | 8–32 | 取决于 GPU 显存 |
| `learning_rate` | 1e-4 | AdamW 优化器 |
| `offline_steps` | 50000 | mock 数据验证用 10000 步即可 |
| `obs_horizon` | 1 | 观测窗口（当前仅关节状态） |

## 4. GPU 资源预估

| 数据规模 | GPU 显存需求 | 训练时间 |
|----------|-------------|----------|
| mock 数据 (100 episodes, state-only) | ~2 GB | ~30 min / 10000 steps |
| 真实数据 (500 episodes, state-only) | ~4 GB | ~2 h / 50000 steps |
| 真实数据 + 图像 (500 episodes) | 12–24 GB | 6–12 h |

## 5. 当前状态

- [x] LeRobot v3.0 转换器完整（data + metadata + stats）
- [x] Mock 数据端到端管道可用（录制 → 转换 → 验证）
- [x] Docker 镜像包含 PyTorch 2.10 + LeRobot 0.4.4 + CUDA 12.9
- [ ] 多 episode mock 数据集（50–100 episodes）
- [ ] ACT 训练回路验证（mock 数据 + 10000 steps）
- [ ] 真实遥操作数据采集

## 6. 注意事项

- **数据量**：ACT 需要 50+ episodes 才能学到有意义的策略。mock 数据用于验证训练回路，真实数据用于训练可用策略。
- **归一化**：stats.json 中的统计量用于训练时归一化输入。确保 stats 基于足够的 episode 计算。
- **FPS 一致性**：录制和训练的 FPS 需一致（默认 30Hz）。如果录制帧率不稳，转换时帧会按时间戳对齐。
- **无图像模式**：当前 state-only 模式只能学到关节空间的运动模式。对于需要视觉反馈的任务（如精确定位、避障），必须添加相机图像。
