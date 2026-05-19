#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/restart-tee.sh [options]

Options:
  --no-clean            Skip make clean before make image.
  --no-build            Skip make clean/make image and only restart.
  --no-health           Skip make health after start.
  --min-free-gb GB      Require at least this much free disk before build/run. Default: 20.
  --follow              Tail nohup.out after starting.
  -h, --help            Show this help.

Examples:
  scripts/restart-tee.sh
  scripts/restart-tee.sh --follow
  scripts/restart-tee.sh --no-build
USAGE
}

DO_CLEAN=1
DO_BUILD=1
DO_HEALTH=1
FOLLOW_LOGS=0
MIN_FREE_GB=20

while [ "$#" -gt 0 ]; do
  case "$1" in
    --no-clean)
      DO_CLEAN=0
      shift
      ;;
    --no-build)
      DO_BUILD=0
      shift
      ;;
    --no-health)
      DO_HEALTH=0
      shift
      ;;
    --min-free-gb)
      MIN_FREE_GB="${2:-}"
      if ! [[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]]; then
        echo "Missing or invalid value for --min-free-gb" >&2
        exit 2
      fi
      shift 2
      ;;
    --follow)
      FOLLOW_LOGS=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if REPO_DIR="$(git -C "$PWD" rev-parse --show-toplevel 2>/dev/null)"; then
  :
elif REPO_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
elif REPO_DIR="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  echo "Could not find a git repository." >&2
  echo "Run this script from inside the TEE repo, or place it in the repo root/scripts directory." >&2
  exit 1
fi

cd "$REPO_DIR"

log() {
  printf '\n[restart-tee] %s\n' "$*"
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

kill_listening_port() {
  local port="$1"
  local pids

  pids="$(sudo lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    return 0
  fi

  log "Killing process(es) listening on port $port: $pids"
  sudo kill $pids 2>/dev/null || true
  sleep 2

  pids="$(sudo lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    log "Force killing process(es) still listening on port $port: $pids"
    sudo kill -9 $pids 2>/dev/null || true
  fi
}

require_free_disk() {
  local path="$1"
  local min_gb="$2"
  local available_kb
  local required_kb

  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  required_kb="$((min_gb * 1024 * 1024))"

  if [ "$available_kb" -lt "$required_kb" ]; then
    echo "Not enough free disk on $(df -Pk "$path" | awk 'NR == 2 {print $1}')." >&2
    echo "Available: $((available_kb / 1024 / 1024)) GB; required: ${min_gb} GB." >&2
    echo "Try: docker system prune -af && docker builder prune -af" >&2
    exit 1
  fi
}

require_cmd git
require_cmd make
require_cmd nitro-cli
require_cmd sudo
require_cmd lsof

log "Repo: $REPO_DIR"

log "Checking free disk space"
require_free_disk "$REPO_DIR" "$MIN_FREE_GB"

log "Current git state"
git status --short --branch

log "Terminating existing Nitro enclaves"
sudo nitro-cli terminate-enclave --all || true

log "Removing stale gvproxy socket"
sudo rm -f /tmp/network.sock

log "Stopping stale port forwarders"
kill_listening_port 2222
kill_listening_port 8000
kill_listening_port 443

if [ -f nohup.out ]; then
  timestamp="$(date +%Y%m%d-%H%M%S)"
  log "Archiving previous nohup.out to nohup.out.$timestamp"
  mv nohup.out "nohup.out.$timestamp"
fi

if [ "$DO_BUILD" -eq 1 ]; then
  if [ "$DO_CLEAN" -eq 1 ]; then
    log "Cleaning previous build artifacts"
    make clean
  fi

  log "Building enclave image"
  make image
fi

log "Checking free disk space before Docker load / EIF build"
require_free_disk "$REPO_DIR" "$MIN_FREE_GB"

log "Starting enclave in background"
nohup make run > nohup.out 2>&1 &
run_pid="$!"
log "Started nohup make run as PID $run_pid"

if [ "$DO_HEALTH" -eq 1 ]; then
  log "Waiting for health endpoint"
  healthy=0
  for attempt in $(seq 1 60); do
    if make health >/tmp/tee-health.out 2>&1; then
      healthy=1
      log "Health check passed on attempt $attempt"
      cat /tmp/tee-health.out
      break
    fi
    sleep 5
  done

  if [ "$healthy" -eq 0 ]; then
    log "Health check did not pass. Last health output:"
    cat /tmp/tee-health.out 2>/dev/null || true
    log "Last 120 lines from nohup.out:"
    tail -n 120 nohup.out || true
    exit 1
  fi
fi

log "Current enclave state"
nitro-cli describe-enclaves || true

log "Restart complete. Logs: $REPO_DIR/nohup.out"

if [ "$FOLLOW_LOGS" -eq 1 ]; then
  tail -f nohup.out
fi
