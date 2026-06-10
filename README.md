# UMI — Universal Manipulation Interface

通用机器人操控接口：基于 Quest 3 VR 手部追踪的遥操作与模仿学习系统。

## 项目状态

| 阶段 | 时间 | 内容 | 状态 |
|------|------|------|------|
| Stage 1 | 月 1–3 | 硬件集成与遥操作回路 | ✅ 完成（35/36 测试通过） |
| Stage 2 | 月 3–6 | 数据采集流水线 | ✅ 完成（仿真、v3.0 转换、质量过滤、离线渲染） |
| Stage 3 | 月 6–10 | 模型训练基线 | ✅ 完成（ACT + DP + 目标条件化 + 视觉特征） |
| Stage 4 | 月 10+ | 部署与泛化 | 🔲 待开始 |

### 当前性能基线

| 指标 | 实测值 | 目标 | 状态 |
|------|--------|------|------|
| 端到端延迟 | p50: 8.1ms | <50ms | ✅ |
| IK 求解 | mean: 26ms, p99: 32ms | <33ms (30Hz) | ✅ |
| 时间同步 | 687μs | <5ms | ✅ |
| FK 精度（MuJoCo vs 手写） | 0.000mm | — | ✅ |
| 离线渲染 | 64×64 RGB, MuJoCo viewer offscreen | — | ✅ (Mesa llvmpipe) |
| 目标条件化到达（8→5 rad） | 38.6% 改善, 0% 泛化差距 | — | 📊 |

### 已注册模型（13 个）

#### 基础模型

| 模型 | 类型 | 参数 | 数据 | Loss | Test MSE |
|------|------|------|------|------|----------|
| act_state_only | ACT | 9.8M | 300 eps 合成 | 0.184 | 0.851 |
| act_diverse_20k | ACT | 9.8M | 300 eps 合成 | 0.154 | 0.90 |
| act_diverse_v2 | ACT | 9.8M | 400 eps 噪声合成 | 0.223 | 0.89 |
| act_teleop | ACT | 9.8M | 11 eps 人类遥操作 | 0.124 | 1.65 |
| act_mixed | ACT | 9.8M | 300 合成 + 11 人类 | 0.199 | 1.77 |
| dp_diverse_10k | DP | 63.3M | 300 eps 合成 | 0.039 | 0.052 |

#### 视觉特征模型

| 模型 | 类型 | 参数 | 输入 | Loss |
|------|------|------|------|------|
| act_visual | ACT | 9.8M | state(6) + image_features(128) | 0.184 |
| dp_visual | DP | 63.3M | state(6) + image_features(128) | 0.053 |

> 使用 `SyntheticFeatureGenerator`（随机傅里叶特征）从关节状态生成 128 维合成视觉特征，模拟冻结视觉编码器输出。state-only ≈ state+visual 证明管道正确。真实图像可无缝替换。

#### 目标条件化模型

| 模型 | 类型 | 参数 | 输入 | 改善 |
|------|------|------|------|------|
| act_multitask | ACT | 9.8M | state(6) + task_index(5) | ❌ 无效 |
| act_goal | ACT | 9.8M | state(6) + goal_position(6) | 12.7% |
| act_goal_la | ACT | 9.8M | state(6) + goal(6), K=8-15 | 26.4% |
| **act_goal_la_big** | **ACT** | **9.8M** | **state(6) + goal(6), K=20-30** | **38.6%** |
| dp_goal_la | DP | 63.3M | state(6) + goal(6) | 训练成功, 推理待修复 |

查看模型：`python3 stage_2/umi_pipeline.py models`

## 目标条件化到达（核心成果）

### 方法演进

| 迭代 | 方法 | 问题 | 改善 |
|------|------|------|------|
| V1 | task_index（one-hot） | 模型忽略离散任务 ID | ❌ 0% |
| V2 | goal_position（连续向量） | 空间信息可用 | 12.7% |
| V3 | + 前瞻动作 K=8-15 | obs-act 差异 0.08→0.39 rad | 26.4% |
| V4 | + 前瞻动作 K=20-30 | obs-act 差异 0.08→0.54 rad | **38.6%** |
| V5 | + 物理步数修复 | 16×mj_step/控制周期 | 训练匹配现实 |

### 系统性评估结果

| 模型 | 平均改善 | 已见目标 | 未见目标 | 泛化差距 |
|------|----------|----------|----------|----------|
| act_state_only (基线) | 4.6% | 4.6% | 4.5% | +0.1% |
| act_goal (K=0) | 12.7% | 13.4% | 11.3% | +2.1% |
| act_goal_la (K=8-15) | 26.4% | 24.9% | 29.3% | −4.4% |
| **act_goal_la_big (K=20-30)** | **38.6%** | **38.4%** | **39.1%** | **−0.7%** |

> **关键结论**：
> 1. 连续 `goal_position` 条件化有效，离散 `task_index` 完全无效
> 2. 前瞻动作（lookahead）是最关键的改进：K=0→K=20-30 带来 3× 提升
> 3. 零泛化差距：未见目标改善率 ≈ 已见目标，证明学到了真正的空间泛化
> 4. 比 state-only 基线好 **8.4×**

## 快速开始（Docker）

```bash
# 1. 构建 Docker 镜像（首次 10-15 分钟）
make build

# 2. 进入容器
make shell

# 3. 编译 ROS2 包（容器内）
cd /ros2_ws && colcon build
```

详细启动指南见 [`docs/START.md`](docs/START.md)。

## 仿真遥操作（键盘控制 + 3D 可视化）

无需硬件，在 MuJoCo 仿真中操控 xArm6 机械臂，支持实时录制。

```bash
make up

# 终端 1：启动控制管道
make exec cmd="bash /workspace/umi/sim_start_pipeline.sh &"

# 终端 2：3D 可视化
make exec cmd="bash /workspace/umi/sim_viewer.sh"

# 终端 3：键盘操控
make exec cmd="bash /workspace/umi/sim_teleop.sh"
```

键盘控制：W/S 前后、A/D 左右、Q/E 升降、I/K 俯仰、J/L 横滚、U/O 偏航、空格夹爪、R 复位、Tab 录制

## 模型训练（完整命令参考）

### 数据生成

```bash
# 基础多样化数据（3 种轨迹类型）
python3 stage_2/generate_diverse_data.py -n 300 -o data/my_dataset --v3

# 目标条件化数据（8 个目标，含前瞻动作 + 物理修正）
python3 stage_2/generate_goal_data.py -n 400 -o data/goal_dataset --v3

# 目标条件化 + 相机图像渲染
python3 stage_2/generate_goal_data.py -n 400 -o data/goal_img_dataset --v3 \
    --render --img-size 64
```

### 训练

```bash
# 基础 ACT
python3 stage_2/train_act.py --data data/dataset_v3 --steps 10000 --batch-size 64

# 目标条件化 ACT
python3 stage_2/train_goal_act.py --data data/goal_dataset_v3 --steps 15000

# 目标条件化 DP
python3 stage_2/train_goal_dp.py --data data/goal_dataset_v3 --steps 10000

# State-only 对比（禁用视觉特征）
python3 stage_2/train_act.py --data data/dataset_v3 --no-visual
```

### 评估

```bash
# 系统性 rollout 评估（所有模型，多目标）
python3 stage_2/evaluate_all_models.py --runs 15

# Pipeline 入口
python3 stage_2/umi_pipeline.py validate    # 快速验证
python3 stage_2/umi_pipeline.py train       # 完整训练
python3 stage_2/umi_pipeline.py compare     # 模型对比
python3 stage_2/umi_pipeline.py models      # 查看注册模型
```

## 架构

### 数据流

```
┌─────────────────────────────────────────────────────────────┐
│ 数据生成层                                                    │
│   generate_diverse_data.py  ───  随机关节空间轨迹              │
│   generate_goal_data.py     ───  目标条件化到达                │
│   generate_multitask_data.py ───  多任务条件化                 │
│   mujoco_renderer.py        ───  离线相机渲染 (64×64 RGB)      │
└──────────────────┬──────────────────────────────────────────┘
                   │ HDF5 (episode_{i:06d}.h5)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 格式转换层                                                    │
│   lerobot_v3_converter.py  ───  LeRobot v3.0 Parquet + JSON  │
│   Features: observation.joint_position, .goal_position,       │
│             .image_features, .image_features, action.*        │
└──────────────────┬──────────────────────────────────────────┘
                   │ Parquet
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 模型训练层                                                    │
│   train_act.py          ───  ACT (9.8M, state-only/baseline) │
│   train_goal_act.py     ───  ACT goal-conditioned            │
│   train_dp.py           ───  Diffusion Policy (63.3M)        │
│   train_goal_dp.py      ───  DP goal-conditioned             │
│   train_multitask_act.py ───  ACT task-conditioned           │
└──────────────────┬──────────────────────────────────────────┘
                   │ checkpoint.pt
                   ▼
┌─────────────────────────────────────────────────────────────┐
│ 评估层                                                        │
│   evaluate_all_models.py ───  系统性 rollout (6 targets)      │
│   evaluate_multitask.py  ───  多任务评估                      │
│   compare_models.py      ───  Test MSE 对比                  │
│   Model Registry         ───  outputs/model_registry.json    │
└─────────────────────────────────────────────────────────────┘
```

### 遥操作实时管道

```
键盘 / Quest3 手部追踪
    │ /hand/right/wrist_pose
    ▼
teleop_bridge (hand_mapper)     ← IK（手写 DH + 雅可比伪逆 + 重复求解跳过）
    │ /teleop/command/joints
    ▼
safety (safety_guardian)        ← 关节限位/速度/急停（mock/xarm6/mujoco 后端）
    │
    ▼
robot_hal (RobotInterface ABC)  ← Mock / xArm6 / MuJoCo 实现
    │
    ▼
recorder (HDF5 录制)            ← 同步多模态数据（关节+夹爪+可选相机）
    │
    ▼
LeRobot v3.0 (Parquet + JSON)   ← lerobot_v3_converter.py
    │
    ▼
ACT / Diffusion Policy          ← 端到端策略训练
    │
    ▼
Model Registry                  ← 元数据 + 评估结果
```

### 模块清单

| 模块 | 路径 | 说明 |
|------|------|------|
| `kinematics` | `stage_1/kinematics/` | FK/IK/Jacobian（纯 Python，不依赖 ROS） |
| `robot_hal` | `stage_1/robot_hal/` | 硬件抽象层（Mock/XArm6/MuJoCo） |
| `teleop_bridge` | `stage_1/teleop_bridge/` | Quest3 → ROS2 桥接 + 手眼标定 + IK 震荡抑制 |
| `perception` | `stage_1/perception/` | USB 相机驱动 |
| `safety` | `stage_1/safety/` | 安全守护节点（mock/xarm6/mujoco 后端） |
| `recorder` | `stage_1/recorder/` | HDF5 录制 + LeRobot v2.x/v3.0 转换 |
| `launch` | `stage_1/launch/` | 全系统启动文件 |
| `stage_2` | `stage_2/` | 数据管道 + MuJoCo 仿真 + 训练 + 评估 |

### Stage 2 完整工具清单

| 类别 | 工具 | 功能 |
|------|------|------|
| 数据 | `generate_diverse_data.py` | 随机关节空间轨迹（3 类型, 速度限制, 噪声） |
| 数据 | `generate_goal_data.py` | 目标条件化到达数据（8 目标, 前瞻动作, 可选渲染） |
| 数据 | `generate_multitask_data.py` | 多任务条件化数据（5 任务, task_index 标签） |
| 渲染 | `mujoco_renderer.py` | 离线 MuJoCo 相机渲染（mujoco-python-viewer 封装） |
| 转换 | `lerobot_v3_converter.py` | HDF5 → LeRobot v3.0 Parquet |
| 训练 | `train_act.py` | ACT 训练（支持 `--no-visual` state-only 消融） |
| 训练 | `train_goal_act.py` | 目标条件化 ACT 训练 |
| 训练 | `train_dp.py` | Diffusion Policy 训练 |
| 训练 | `train_goal_dp.py` | 目标条件化 DP 训练 |
| 训练 | `train_multitask_act.py` | 多任务 ACT 训练 |
| 评估 | `evaluate_all_models.py` | 系统性 rollout 评估（6 目标, 统计指标） |
| 评估 | `evaluate_multitask.py` | 多任务/目标条件化策略评估 |
| 评估 | `evaluate_rollout.py` | 基础 rollout 评估 |
| 评估 | `compare_models.py` | 多模型 Test MSE 对比 |
| 管道 | `umi_pipeline.py` | 统一 CLI 入口（validate/train/compare/models） |
| 管道 | `validate_pipeline.py` | 端到端小规模完整性检查 |
| 质量 | `quality_filters.py` | 9 项数据质量检查 |
| 质量 | `quality_check.py` | 录制 + 质量分析端到端测试 |
| 任务 | `task_manager.py` | 任务定义 + tasks.parquet 导出 |
| 诊断 | `time_sync_check.py` | 时间同步精度评估 |
| 诊断 | `ik_diagnostic.py` | IK 收敛性诊断 |
| 研究 | `act_research.py` | ACT 基线研究（批量录制 + v3 + 验证） |
| 仿真 | `simulation/keyboard_teleop.py` | 键盘遥操作（WASD, Tab 录制） |
| 仿真 | `simulation/viewer_node.py` | MuJoCo 3D 可视化 |
| 仿真 | `simulation/mujoco_interface.py` | MuJoCo RobotInterface 实现 |
| 仿真 | `simulation/xarm6.xml` | 精确 DH MJCF 模型（FK 误差 0mm） |
| 仿真 | `simulation/xarm_color/` | STL 网格彩色模型 |

## 关键发现与技术决策

### 目标条件化
- **task_index（离散 one-hot）无效**：模型输出与 task_index 完全无关（|pred_1 − pred_2| = 0）
- **goal_position（连续向量）有效**：6 维空间目标提供可用的梯度信号
- **泛化差距为零**：已见 38.4% ≈ 未见 39.1%，证明学到了真正的空间映射

### 前瞻动作（Lookahead Actions）
- 核心问题：`obs − act` 差异仅 ~0.08 rad，模型学到"保持不动"
- 解决方案：`action[i] = joint_command[i + K]`（K=20-30 步超前）
- 效果：差异增大 7×（0.54 rad），改善提升 3×（13% → 39%）

### MuJoCo 物理步数
- 每个控制周期（30Hz）需 16 次 `mj_step()`（dt=0.002s）
- 之前只调用 1 次，伺服无法在 0.002s 内到达目标
- 修复后模型在仿真中的运动更真实

### 离线渲染
- 历时数天调试 EGL/OSMesa/GLX 均失败（全部产生黑色图像）
- 最终方案：使用 `mujoco-python-viewer` 的 offscreen 模式
- 64×64 RGB 图像可正常渲染，存储为 `sensors/camera/rgb`
- Mesa llvmpipe 软件渲染可用，无需 GPU 驱动

## 测试

```bash
# 纯 Python 测试（宿主机）：
PYTHONPATH=. python3 -m pytest stage_1/tests/test_fk.py stage_1/tests/test_ik.py \
       stage_1/tests/test_jacobian.py stage_1/tests/test_mock_robot.py \
       stage_1/tests/test_hdf5_writer.py stage_1/tests/test_lerobot_converter.py -v

# 完整 ROS2 集成测试（容器内）：
make shell cmd="bash -c 'cd /ros2_ws && colcon build && \
  source install/setup.bash && export RMW_IMPLEMENTATION=rmw_fastrtps_cpp && \
  python3 -m pytest /workspace/umi/stage_1/tests/ -v \
  -p no:launch_testing -p no:launch_testing_ros'"
```

## 规划

- 项目整体规划：[`planning/full_planning/full_v1.md`](planning/full_planning/full_v1.md)
- 每日任务记录：`planning/tomorrow_*.md`
- 开发指南：[`CLAUDE.md`](CLAUDE.md)
- 阶段一文档：[`stage_1/README.md`](stage_1/README.md)

## 技术栈

- **ROS2 Humble** (Ubuntu 22.04) · **Python 3.10** · **Docker**
- **MuJoCo** 物理仿真 + 离线渲染 · **mujoco-python-viewer**
- **LeRobot** (HuggingFace) · Parquet v3.0 格式 + ACT/DP 策略 API
- **ACT** (Action Chunking Transformer) 9.8M · **Diffusion Policy** (UNet) 63.3M
- **Quest 3** VR 手部追踪 · **xArm6** 机械臂
- **PyTorch** CUDA 12.1 · NVIDIA RTX 4060 (7.6GB)

## 许可证

MIT
