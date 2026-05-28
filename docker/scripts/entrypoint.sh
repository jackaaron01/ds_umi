#!/bin/bash
# =============================================================================
# UMI 容器入口脚本
# =============================================================================
# 这个脚本在每次容器启动时执行，负责初始化 ROS2 环境。
#
# 执行流程：
#   1. 将 ROS2 环境变量注入 ~/.bashrc（让每个终端自动生效）
#   2. 创建 workspaces 的符号链接（让 ROS2 包可以被 colcon 找到）
#   3. 执行传入的命令（默认为 /bin/bash）
#
# 为什么需要这个脚本？
#   - ROS2 依赖大量环境变量（AMENT_PREFIX_PATH、LD_LIBRARY_PATH 等）
#   - 这些变量通过 setup.bash 设置，不在 .bashrc 中则每个新 shell 需手动 source
#   - entrypoint 自动化这个过程，保证进入容器即可用 ROS2 命令
# =============================================================================

set -e

# ---------------------------------------------------------------------------
# ROS2 环境初始化
# ---------------------------------------------------------------------------
# setup.bash 做了什么？
#   - 设置 AMENT_PREFIX_PATH：colcon/aement 用它找 ROS2 包
#   - 设置 PYTHONPATH：Python import 能找到 ROS2 的 Python 包
#   - 设置 LD_LIBRARY_PATH：动态链接器能找到 ROS2 的 .so 库
#   - 设置 ROS_DISTRO、ROS_VERSION 等环境变量
#   - 注册 ros2 命令行工具的 bash 补全

ROS2_SETUP="/opt/ros/humble/setup.bash"
ROS2_WS_SETUP="/ros2_ws/install/setup.bash"

# 将 source 命令写入 ~/.bashrc（仅当尚未写入时）
# 为什么要写入 .bashrc 而不是直接 source？
#   - entrypoint.sh 中的 source 只对当前进程有效
#   - 后续打开的新终端（或 docker exec 进入的 shell）需要重新 source
#   - 写入 .bashrc 确保每个交互式 shell 都能使用 ROS2 命令
if ! grep -q "source ${ROS2_SETUP}" ~/.bashrc 2>/dev/null; then
    echo "source ${ROS2_SETUP}" >> ~/.bashrc
    echo "# ROS2 Humble environment sourced by entrypoint.sh" >> ~/.bashrc
fi

# 工作空间 setup（如果已经 colcon build 过则会生成这个文件）
if [ -f "${ROS2_WS_SETUP}" ]; then
    if ! grep -q "source ${ROS2_WS_SETUP}" ~/.bashrc 2>/dev/null; then
        echo "source ${ROS2_WS_SETUP}" >> ~/.bashrc
        echo "# UMI colcon workspace overlay" >> ~/.bashrc
    fi
fi

# ---------------------------------------------------------------------------
# 将项目中的 ROS2 包链接到 colcon 工作空间
# ---------------------------------------------------------------------------
# 为什么需要符号链接？
#   - colcon 在 /ros2_ws/src/ 下找包并编译
#   - 我们的源码在 /workspace/umi/ 下（通过 volume 挂载）
#   - 通过 symlink，colcon 可以看到源码包，但编译产物放在 /ros2_ws/build/ 和 install/
#   - 源码修改后只需重新 colcon build，不需要复制文件
#
# 注意：只在符号链接不存在时创建，避免重复创建和权限错误
WS_SRC="/ros2_ws/src"
PROJECT_ROOT="/workspace/umi"

# ROS2 包的目录列表（含有 package.xml 的目录）
# 这些是 colcon 需要编译的包，不包括纯 Python 包（robot_hal/ kinematics/）
ROS2_PACKAGES=(
    "teleop_bridge"
    "perception"
    "safety"
    "recorder"
    "launch"
)

for pkg in "${ROS2_PACKAGES[@]}"; do
    if [ -d "${PROJECT_ROOT}/${pkg}" ] && [ ! -L "${WS_SRC}/${pkg}" ]; then
        ln -s "${PROJECT_ROOT}/${pkg}" "${WS_SRC}/${pkg}" 2>/dev/null || true
    fi
done

# ---------------------------------------------------------------------------
# 在当前 shell 中立即 source ROS2 环境
# ---------------------------------------------------------------------------
# 上面的 .bashrc 写入保证了交互式 shell 中可用，但非交互式 shell
# （如 docker compose run umi-dev ros2 topic list）不会读取 .bashrc。
# 所以这里必须直接 source，让后续的 exec "$@" 继承环境变量。
source ${ROS2_SETUP}

if [ -f "${ROS2_WS_SETUP}" ]; then
    source ${ROS2_WS_SETUP}
fi

echo "[UMI entrypoint] ROS2 environment ready."
echo "[UMI entrypoint] Project: ${PROJECT_ROOT}"
echo "[UMI entrypoint] Colcon workspace: /ros2_ws"

# ---------------------------------------------------------------------------
# 执行传入的命令
# ---------------------------------------------------------------------------
# exec "$@" : 用传入的命令替换当前 shell 进程
#   - 如果 CMD 是 /bin/bash → 进入交互式 shell
#   - 如果 docker compose run umi-dev ros2 topic list → 执行该命令然后退出
#   - 使用 exec（而非直接运行）让传入的命令成为 PID 1，信号处理更正确
exec "$@"
