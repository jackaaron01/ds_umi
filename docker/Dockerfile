# =============================================================================
# UMI (Universal Manipulation Interface) 开发环境镜像
# =============================================================================
# 本文件定义了整个项目的运行环境，所有依赖都封装在容器内，不会污染宿主机。
#
# 设计原则：
#   1. 可复现：任何人拿到这个 Dockerfile 都能构建出完全相同的环境
#   2. 分层清晰：变动频率低的层放上面（系统包），变动频率高的放下面（Python 包）
#   3. 学习导向：每个选择都有明确的原因，见各层注释
# =============================================================================

# ============================================================================
# 第 1 层：基础镜像选择
# ============================================================================
# 为什么选 osrf/ros:humble-desktop-full？
#   - osrf = Open Source Robotics Foundation，ROS 官方维护的 Docker 镜像
#   - humble = ROS2 Humble Hawksbill，2022 年 5 月发布的 LTS 版本，支持到 2027 年 5 月
#   - desktop-full = 包含 RViz2（可视化）、Gazebo（仿真）、ros2_control 等全套桌面工具
#   - 底层是 Ubuntu 22.04（Jammy），与你的宿主机一致，内核 ABI 完全兼容
#
# 备选方案对比：
#   - ros:humble-ros-base：只有 ROS2 核心，没有 GUI 工具，不适合需要 RViz 的开发
#   - ros:jazzy-desktop-full：2024 年新 LTS，但基于 Ubuntu 24.04，与 22.04 宿主机有差异
#   - nvidia/cuda:xxx + 手动装 ROS2：过于复杂，OSRF 镜像已经做好 ROS2 集成
FROM osrf/ros:humble-desktop-full

# ============================================================================
# 第 2 层：构建参数（ARG）—— 用户身份配置
# ============================================================================
# 为什么需要这些 ARG？
#   - Docker 容器默认以 root 运行，但容器内创建的文件会在宿主机上显示为 root 所有
#   - 通过传入宿主机的 UID/GID，容器内用户和宿主机用户拥有相同的 ID
#   - 这样你在容器内创建/修改的文件，回到宿主机不会遇到权限问题
#   - 原理：Linux 文件权限基于数字 UID/GID，而非用户名。容器内 UID=1000 的用户
#     创建的文件，在宿主机上 UID=1000 的用户可以直接读写
#
# 默认值 1000:1000 是 Ubuntu 桌面系统安装时创建的第一个用户的典型 ID
ARG USER_UID=1000
ARG USER_GID=1000
ARG USER_NAME=dev

# ============================================================================
# 第 3 层：系统级依赖 —— apt 包
# ============================================================================
# 每一个包都有它的存在理由，不是随便装的：
#
# 编译与构建工具：
#   build-essential : gcc/g++/make 等，编译 C++ 代码（Pinocchio 的依赖可能需要）
#
# 开发便利工具：
#   git             : 版本控制
#   vim             : 容器内编辑文件（你也可以装 nano 或 emacs）
#   curl wget       : 下载文件
#   bash-completion : ROS2 命令行的 Tab 补全
#
# Python 相关：
#   python3-pip     : pip 包管理器
#   python3-dev     : Python 头文件，编译 C 扩展时需要（如 h5py）
#   python3-tk      : matplotlib 等需要 Tk 后端的场景
#
# HDF5 数据格式（阶段一录制数据用）：
#   libhdf5-dev     : HDF5 C 库的开发头文件
#   hdf5-tools      : h5ls、h5dump 等命令行工具，调试 HDF5 文件
#
# 图形与显示（RViz / Gazebo / matplotlib 绘图）：
#   libgl1-mesa-glx : OpenGL 软件渲染（没有它 RViz 会 crash）
#   libglib2.0-0    : GTK/Qt 事件循环依赖
#   libxcb-*        : X11 协议库，Qt 应用（RViz）需要它们与 X Server 通信
#   mesa-utils      : glxinfo 等诊断工具，排查 GPU/显示问题时用
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    curl \
    wget \
    bash-completion \
    python3-pip \
    python3-dev \
    python3-tk \
    libhdf5-dev \
    hdf5-tools \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libxcb-cursor0 \
    libxcb-icccm4 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-xkb1 \
    libxkbcommon-x11-0 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# 第 4 层：ROS2 附加包
# ============================================================================
# humble-desktop-full 已经包含了很多东西，但缺少一些我们需要的：
#
#   ros-humble-ros2-control      : ros2_control 框架，标准化硬件接口
#     - 提供 controller_manager、hardware_interface 等核心抽象
#     - 与我们的 robot_hal/ 层互补：ros2_control 管理控制器生命周期，
#       robot_hal/ 提供跨机器人的统一 Python 接口
#
#   ros-humble-ros2-controllers  : 常用控制器实现
#     - joint_trajectory_controller : 关节空间轨迹跟踪
#     - forward_command_controller  : 直接位置/速度/力矩前馈
#     - 调试时可以直接用，不需要自己写控制器
#
#   ros-humble-joint-state-publisher : 发布关节状态，RViz 可视化需要
#
#   ros-humble-xacro             : URDF/XACRO 宏处理器
#     - xArm6 的 URDF 描述文件通常用 xacro 格式，需要编译成纯 URDF
#
#   ros-humble-plotjuggler       : 时间序列可视化工具
#     - 调试关节轨迹、检查延迟问题时非常有用
#     - 可以直接订阅 ROS2 topic 实时绘图
#
#   ros-humble-rosbag2           : 录制/回放 ROS2 消息
#     - 原生的数据录制工具，用于快速验证数据流
#     - 正式的高质量录制用 recorder/ 模块，rosbag2 用于调试
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-joint-state-publisher \
    ros-humble-xacro \
    ros-humble-plotjuggler \
    ros-humble-rosbag2 \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# 第 5 层：Pinocchio —— 刚体动力学库
# ============================================================================
# Pinocchio 是什么？
#   - 一个高性能 C++ 刚体动力学库，带 Python 绑定
#   - 核心功能：Forward Kinematics (FK)、Inverse Dynamics、Jacobian 计算
#   - 比我们自己手写快 10-100 倍（SIMD 优化、高效的递归算法）
#
# 在我们的项目中的角色：
#   - kinematics/ 模块手写 FK/IK 用于学习理解
#   - Pinocchio 作为"参考答案"验证手写实现的正确性
#   - 后续性能关键路径（如模型训练的在线 IK 求解）可以直接用 Pinocchio
#
# 安装方式：apt install ros-humble-pinocchio
#   - ROS2 官方 apt 仓库直接提供预编译的 Pinocchio 包
#   - 包含 C++ 头文件/库文件和 Python 3.10 绑定
#   - robotpkg 已不再支持 Ubuntu 22.04（jammy），所以用 ROS2 自己的包最可靠
#   - 同时安装 kinematics-interface-pinocchio（ros2_control 的 Pinocchio 插件）
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-pinocchio \
    ros-humble-kinematics-interface-pinocchio \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# 第 6 层：Python 包 —— 核心依赖
# ============================================================================
# 先升级 pip，ROS2 Humble 自带的 pip 版本较旧
RUN python3 -m pip install --upgrade pip

# --- 科学计算基础包 ---
# numpy  : 矩阵运算、线性代数，整个机器人计算的基石
# scipy  : 优化算法（IK 求解中的数值优化）、插值（轨迹平滑）
# h5py   : HDF5 文件的 Python 读写接口（录制数据）
# matplotlib : 调试时画关节轨迹图、误差曲线
# pytest : 单元测试框架
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    scipy \
    h5py \
    matplotlib \
    pytest \
    pandas \
    pyarrow \
    opencv-python-headless

# --- PyTorch（深度学习框架）---
# 阶段一暂不需要训练模型，但预装的好处：
#   1. 提前验证 GPU 在容器内可用
#   2. 后续阶段无需修改 Dockerfile 重新构建
#   3. 可以写简单的推理脚本测试数据加载
#
# --index-url 指定 CUDA 12.1 的预编译包：
#   宿主机的 CUDA 13.2 驱动是向后兼容的，可以运行 CUDA 12.1 编译的程序
#   PyTorch 自带 CUDA 运行时库，不依赖容器内的 CUDA toolkit
# --ignore-installed: ROS2 基础镜像中通过 apt 预装了旧版 sympy/mpmath，
# PyTorch 需要更新版本的 sympy，直接 pip install 会因 distutils 冲突失败。
# --ignore-installed 强制覆盖系统包，避免冲突。
RUN pip install --no-cache-dir --ignore-installed \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu121

# --- 机器人相关 Python 库 ---
# lerobot           : HuggingFace 维护的机器人学习框架
#                     - 提供标准化的数据集格式（LeRobot format = HDF5 + Parquet）
#                     - 内置多种 BC/ACT/Diffusion Policy 实现
#                     - 我们主要用它的数据格式和 dataset 加载器
#
# hand-tracking-sdk : Meta Quest 手部追踪数据的 Python 解析库
#                     - 解析 Hand Tracking Streamer App 发来的 UDP 数据包
#                     - 输出 21 个手部关键点 + 手腕 6-DoF 位姿
#                     - 自带简单的 3D 可视化
#
# xarm-python-sdk   : 越疆科技官方的 xArm Python SDK
#                     - 提供对 xArm6 的完整控制（关节/笛卡尔/示教模式）
#                     - 基于 Modbus TCP 或 USB 串口通信
#                     - 我们的 XArm6Interface 会封装这个 SDK
RUN pip install --no-cache-dir \
    lerobot \
    hand-tracking-sdk \
    xarm-python-sdk \
    mujoco \
    mujoco-python-viewer

# --- 其他开发工具 ---
# ipython : 增强的 Python 交互式 shell（比 python3 好用得多）
# black   : Python 代码格式化工具（统一代码风格）
# ruff    : 快速的 Python linter（比 flake8 快 10-100 倍）
RUN pip install --no-cache-dir \
    ipython \
    black \
    ruff

# --- 修复 scipy/numpy 兼容性 ---
# PyTorch 的 --ignore-installed 会将 numpy 升级到 2.x，
# 而基础镜像中的 python3-scipy 1.8.0 需要 numpy <1.25。
# 强制安装 scipy>=1.12 以兼容 numpy 2.x。
RUN pip install --no-cache-dir --force-reinstall --ignore-installed "scipy>=1.12"

# ============================================================================
# 第 7 层：创建非 root 用户
# ============================================================================
# 为什么要创建专门的用户而不是用 root？
#   1. 安全：容器内编译/运行代码不应该有 root 权限
#   2. 文件权限：容器内创建的文件属于这个用户，UID 与宿主机一致
#   3. ROS2 最佳实践：rosdep/colcon 在非 root 用户下运行更稳定
#
# groupadd -g $GID : 创建与宿主机相同 GID 的组
# useradd -m       : -m 创建 home 目录（/home/dev）
#        -u $UID   : 指定 UID 与宿主机一致
#        -g $GID   : 指定主组
#        -G sudo   : 附加 sudo 组（偶尔需要安装东西）
#        -s bash   : 默认 shell
RUN groupadd -g ${USER_GID} ${USER_NAME} \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -G sudo -s /bin/bash ${USER_NAME} \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USER_NAME}

# ============================================================================
# 第 8 层：ROS2 工作空间初始化
# ============================================================================
# 创建一个 colcon 工作空间，后续所有 ROS2 包（teleop_bridge/ perception/ 等）放这里编译
#
# colcon 是什么？
#   - ROS2 的构建工具（类比 CMake 之于 C++，但专门为 ROS 包设计）
#   - colcon build 自动处理多个包的依赖关系和编译顺序
#   - 编译产物放在 install/ 目录，source install/setup.bash 后可用
#
# 为什么工作空间放在 /ros2_ws 而不是项目目录内？
#   - 项目源码通过 volume 挂载在 /workspace/umi
#   - colcon 的 build/ install/ log/ 目录很大（编译产物），不应污染源码目录
#   - 分离后 colcon clean 也更方便（直接删 build/ install/）
RUN mkdir -p /ros2_ws/src && chown -R ${USER_NAME}:${USER_NAME} /ros2_ws

# ============================================================================
# 第 9 层：容器入口脚本
# ============================================================================
# entrypoint.sh 在每次容器启动时执行，负责：
#   1. 将 ROS2 的 setup.bash 写入 ~/.bashrc，确保每个新 shell 都能用 ROS2 命令
#   2. source 工作空间的 setup.bash（如果已编译）
#   3. 执行 docker-compose 或 docker run 传来的命令
COPY docker/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ============================================================================
# 第 10 层：切换用户与工作目录
# ============================================================================
# 从现在开始，所有操作以 dev 用户身份执行（不再用 root）
#
# 为什么 WORKDIR 设置为 /workspace/umi？
#   - 这是项目的挂载点，docker-compose.yml 会把宿主机 umi/ 目录挂载到这里
#   - 进入容器后自动处于项目根目录，方便直接操作
USER ${USER_NAME}
WORKDIR /workspace/umi

# ENTRYPOINT vs CMD 的区别：
#   ENTRYPOINT : 容器启动时固定执行的脚本（这里的 entrypoint.sh）
#   CMD        : 默认参数，可以被 docker run 后面的命令覆盖
#   所以 "docker compose run umi-dev bash" → entrypoint.sh 收到 "bash" 作为参数
#       "docker compose run umi-dev"      → entrypoint.sh 收到 "/bin/bash" 作为参数
ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
