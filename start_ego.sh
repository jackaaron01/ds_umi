#!/bin/bash
# =============================================================================
# EGO Teleop — 一键启动（Docker 仿真器 + 提示宿主机追踪命令）
#
# 用法: bash start_ego.sh
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE="docker compose --project-directory ${SCRIPT_DIR}/docker"
CONTAINER="umi-dev"
RESTART="/workspace/umi/stage_2/ego/scripts/restart_ego_sim.sh"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
log()  { echo -e "${GREEN}[ego]${NC} $1"; }
warn() { echo -e "${YELLOW}[ego]${NC} $1"; }
err()  { echo -e "${RED}[ego]${NC} $1"; }

cleanup() {
    echo ""
    log "Stopping simulator..."
    $COMPOSE exec -T "$CONTAINER" pkill -f "mujoco_ego_sim" 2>/dev/null || true
    log "Done."
    exit 0
}
trap cleanup INT TERM

# ── Step 1: Docker container ──────────────────────────────────────────
if $COMPOSE ps --format json 2>/dev/null | grep -q "$CONTAINER"; then
    log "Docker container already running"
else
    log "Starting Docker container..."
    $COMPOSE up -d
    sleep 3
    log "Container started"
fi

# ── Step 2: Simulator ─────────────────────────────────────────────────
log "Starting MuJoCo simulator..."
$COMPOSE exec "$CONTAINER" bash "$RESTART"
log "Simulator launched (window should appear)"

# ── Step 3: Instructions ──────────────────────────────────────────────
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  🎯 Simulator ready — now run in ANOTHER terminal:${NC}"
echo ""
echo -e "  ${GREEN}conda activate ego${NC}"
echo -e "  ${GREEN}cd ${SCRIPT_DIR}${NC}"
echo -e "  ${GREEN}python stage_2/ego/mediapipe_ego.py --udp${NC}"
echo ""
echo -e "  ${YELLOW}Multi-camera (more robust):${NC}"
echo -e "  ${GREEN}python stage_2/ego/mediapipe_ego.py --udp --camera-serials SN1 SN2${NC}"
echo ""
echo -e "${CYAN}  MuJoCo window: 1=Ego  2=Fixed  3/Space=Free   |   q=Quit${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════════${NC}"
echo ""

# Tail simulator log until Ctrl+C
log "Tailing simulator log (Ctrl+C to stop all)..."
$COMPOSE exec "$CONTAINER" tail -f /tmp/ego_sim.log 2>/dev/null || true
