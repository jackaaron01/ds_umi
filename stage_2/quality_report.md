# Stage 2 — 数据质量分析报告

> 基于 mock pipeline 录制 episode 的数据分析
> 录制日期：2026-05-29

## 1. 当前 HDF5 格式 vs LeRobot v3.0

### 当前格式（UMI Stage 1）

```
episode_000000.h5
└── episode_000000/              # HDF5Writer 按 episode 分组
    ├── joint_command/position   (N, 6) float64
    ├── joint_command/position_timestamp  (N, 1) float64
    ├── joint_state/position     (N, 6) float64
    ├── joint_state/position_timestamp    (N, 1) float64
    ├── joint_state/velocity     (N, 6) float64
    ├── joint_state/velocity_timestamp    (N, 1) float64
    ├── gripper/command          (N, 1) float64
    ├── gripper/command_timestamp (N, 1) float64
    ├── gripper/state            (N, 1) float64
    ├── gripper/state_timestamp  (N, 1) float64
    └── sensors/camera/rgb       (N, H, W, 3) uint8  (可选)
```

### LeRobot v3.0 目标格式

LeRobot v3.0 已从 HDF5 迁移到 **Parquet + MP4**:

```
dataset_root/
├── data/
│   └── chunk-000/
│       └── file-000.parquet     # 表格数据（含 observation/action/timestamp）
├── meta/
│   ├── info.json                # 数据集配置（fps, features, robot_type）
│   ├── stats.json               # 归一化统计量
│   ├── tasks.parquet            # task_index → task 描述映射
│   └── episodes/
│       └── chunk-000/
│           └── file-000.parquet # episode 边界元数据
└── videos/
    └── observation.images.laptop/
        └── chunk-000/
            └── file-000.mp4
```

**关键差异：**
- v3.0 不再使用 HDF5，需更新 `lerobot_converter.py` 生成 Parquet + MP4
- v3.0 使用 `info.json`（非 `features.json`）
- v3.0 数据按 chunk 分片（默认 100 文件/chunk）
- 每帧包含 `timestamp`, `frame_index`, `episode_index`, `index`, `task_index`

## 2. Mock Episode 数据质量分析

### 测试条件
- Mock pipeline (mock_hand_tracker → hand_mapper → safety_guardian → recorder)
- 录制时长: 10 秒
- 录制频率目标: 30 Hz

### 2.1 录制统计

| 指标 | 实测值 | 目标值 | 评估 |
|------|--------|--------|------|
| 总步数 | 166 | ~300 (30Hz×10s) | 偏低 |
| 有效频率 | ~17 Hz | 30 Hz | 不达标 |
| 帧间隔均值 | 56.6 ms | 33.3 ms | 偏大 |
| 帧间隔标准差 | 485 ms | <10 ms | 严重抖动 |
| 帧丢失数 | 8 | 0 | 需改进 |

### 2.2 关节指令质量

| 指标 | 值 | 评估 |
|------|-----|------|
| 帧间最大变化 | 0.068 rad (joint 6: 0.135 rad) | 平滑 |
| 帧间平均变化 | 0.0007-0.0014 rad | 非常平滑 |
| 速度尖峰 (>3.14 rad/s) | 1 次 (joint 6) | 可接受 |
| 关节范围 | 全部在限位内 | 正常 |

### 2.3 夹爪质量

| 指标 | 值 | 评估 |
|------|-----|------|
| 取值范围 | [0.621, 0.647] | 范围过窄（mock 合成信号） |
| 越界值 | 0 | 正常 |

### 2.4 指令-状态一致性

| 指标 | 值 | 评估 |
|------|-----|------|
| Joint 1-2 偏差 | ~0.01 rad | 正常（模拟运动延迟） |
| Joint 4 偏差 | ~0.63 rad | 偏大（mock 初始状态为 0） |

## 3. 数据质量过滤规则草稿

### 3.1 完整性检查

```python
MIN_EPISODE_STEPS = 30       # 少于 30 步的 episode 丢弃（<1 秒数据无意义）
MAX_TIMESTAMP_GAP = 0.100    # 相邻帧间隔 > 100ms 视为帧丢失（3× 30Hz 目标间隔）
MAX_FRAME_DROP_RATIO = 0.1   # 帧丢失率 > 10% 标记为低质量
```

### 3.2 运动学检查

```python
MAX_JOINT_VELOCITY = 3.14    # rad/s，xArm6 限速
MAX_JOINT_DELTA = 0.3        # rad，单步最大关节变化
VELOCITY_SPIKE_THRESH = 3.0  # × velocity_limit 内的瞬态可接受
MAX_CONSECUTIVE_SPIKES = 3   # 连续尖峰数 → 标记异常段
```

### 3.3 语义检查

```python
GRIPPER_RANGE = (0.0, 1.0)           # 夹爪必须在 [0,1]
JOINT_LIMIT_MARGIN = 0.05            # 离限位 0.05 rad 以内标记 WARNING
MIN_JOINT_VARIATION = 0.001          # 关节角变化 < 0.001 rad 全程 → 无动作 episode
ZERO_COMMAND_THRESHOLD = 0.9         # > 90% 的帧关节角未变化 → 可能录制失败
```

### 3.4 时间戳检查

```python
MIN_EFFECTIVE_FPS = 10.0             # 有效帧率 < 10 Hz → 丢弃
TIMESTAMP_MONOTONIC = True           # 时间戳应严格递增
MAX_CLOCK_DRIFT = 0.050              # timestamp 与系统时钟偏差 > 50ms → 标记
```

### 3.5 图像检查（如果有相机）

```python
MIN_IMAGE_MEAN = 5.0                 # 平均像素值 < 5 → 可能全黑帧
MAX_IMAGE_MEAN = 250.0               # 平均像素值 > 250 → 可能过曝
IMAGE_VARIANCE_THRESH = 10.0         # 方差 < 10 → 可能无内容
```

## 4. 待办事项

- [ ] 更新 `lerobot_converter.py` 支持 LeRobot v3.0 格式（Parquet + MP4）
- [ ] 将质量过滤规则实现为 `stage_2/quality_filters.py`
- [ ] 提升 recorder 有效录制频率到 30 Hz（调查 ring buffer flush timing）
- [ ] 修复 IK 收敛率问题（当前 mock 模式下频繁失败）
- [ ] 添加 per-step timestamp 录制（当前已有 `*_timestamp` 数据集）
- [ ] 设计数据收集任务（task design）— 参考全规划 Stage 2
