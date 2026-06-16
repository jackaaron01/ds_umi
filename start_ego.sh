#!/bin/bash
# =============================================================================
# EGO Teleop — 一键启动脚本
#
# 自动完成：
#   1. 启动 Docker 容器（如未运行）
#   2. 启动 MuJoCo 仿真器（Docker 内）
#   3. 启动 MediaPipe 手部追踪（宿主机 conda）
#   4. Ctrl+C 一键停止全部
#
# 用法：
#   bash start_ego.sh
#
# 前提：
#   - Docker 镜像已构建 (make build)
#   - conda 环境 "ego" 已配置（含 mediapipe, pyrealsense2, matplotlib）
#   - RealSense D435i 已连接
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DOCKER_COMPOSE="docker compose --project-directory ${SCRIPT_DIR}/docker"
CONTAINER="umi-dev"
SIM_LOG="/tmp/ego_sim.log"
CONDA_ENV="ego"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[ego]${NC} $1"; }
log_warn()  { echo -e "${YELLOW}[ego]${NC} $1"; }
log_err()   { echo -e "${RED}[ego]${NC} $1"; }
log_step()  { echo -e "${CYAN}[ego]${NC} $1"; }

# ── Cleanup on exit ────────────────────────────────────────────────────────
cleanup() {
    echo ""
    log_info "Shutting down..."
    # Kill host-side mediapipe
    if [ -n "$MEDIAPIPE_PID" ] && kill -0 "$MEDIAPIPE_PID" 2>/dev/null; then
        kill "$MEDIAPIPE_PID" 2>/dev/null || true
        log_info "MediaPipe tracking stopped"
    fi
    # Kill simulator in Docker
    $DOCKER_COMPOSE exec -T "$CONTAINER" pkill -f "mujoco_ego_sim" 2>/dev/null || true
    log_info "Simulator stopped"
    log_info "Done."
    exit 0
}
trap cleanup INT TERM

# ── Step 1: Ensure Docker container is running ─────────────────────────────
log_step "Step 1/3: Checking Docker container..."
if $DOCKER_COMPOSE ps --format json 2>/dev/null | grep -q "$CONTAINER"; then
    log_info "Container '$CONTAINER' already running"
else
    log_info "Starting container..."
    $DOCKER_COMPOSE up -d
    sleep 3
    log_info "Container started"
fi

# ── Step 2: Start MuJoCo simulator in Docker ───────────────────────────────
log_step "Step 2/3: Starting MuJoCo simulator..."
# Kill any existing simulator
$DOCKER_COMPOSE exec -T "$CONTAINER" pkill -f "mujoco_ego_sim" 2>/dev/null || true
sleep 1

# Launch simulator in background
$DOCKER_COMPOSE exec -d "$CONTAINER" \
    bash -c 'export PYTHONPATH=/workspace/umi && nohup python3 -u /workspace/umi/stage_2/ego/simulation/mujoco_ego_sim.py --port 9999 > /tmp/ego_sim.log 2>&1 &'

# Wait for simulator to be ready
log_info "Waiting for simulator to initialize..."
for i in $(seq 1 15); do
    sleep 1
    if $DOCKER_COMPOSE exec -T "$CONTAINER" cat /tmp/ego_sim.log 2>/dev/null | grep -q "Running"; then
        log_info "Simulator ready (PID in container)"
        break
    fi
    if [ $i -eq 15 ]; then
        log_warn "Simulator may not be ready yet, continuing..."
        $DOCKER_COMPOSE exec -T "$CONTAINER" cat /tmp/ego_sim.log 2>/dev/null || true
    fi
done

# ── Step 3: Start MediaPipe hand tracking on host ──────────────────────────
log_step "Step 3/3: Starting MediaPipe hand tracking..."

# Check conda env
if ! conda env list 2>/dev/null | grep -q "$CONDA_ENV"; then
    log_err "Conda environment '$CONDA_ENV' not found!"
    log_err "Create it: conda create -n ego python=3.10 && conda activate ego && pip install mediapipe pyrealsense2 matplotlib opencv-python"
    exit 1
fi

# Check RealSense camera
if ! python3 -c "import pyrealsense2 as rs; ctx=rs.context(); print(len(ctx.devices))" 2>/dev/null; then
    log_warn "No RealSense device detected, but continuing anyway..."
fi

log_info "Launching MediaPipe tracker (close window or Ctrl+C to quit)"
echo ""
log_info "══════════════════════════════════════════════════════════"
log_info "  🎯 EGO Teleop running — move hand in front of camera"
log_info "  Press 'q' or Ctrl+C to quit"
log_info "══════════════════════════════════════════════════════════"
echo ""

# Run mediapipe in conda env (foreground)
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
python3 "${SCRIPT_DIR}/stage_2/ego/mediapipe_ego.py" --udp &
MEDIAPIPE_PID=$!

# Wait for mediapipe to finish
wait "$MEDIAPIPE_PID" 2>/dev/null || true
cleanup
