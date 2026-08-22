#!/usr/bin/env bash
set -euo pipefail

# Resolve all paths relative to this script — nothing is hardcoded.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINO_NODE="$SCRIPT_DIR/dino_tracker_node.py"
REFERENCE_NODE="$SCRIPT_DIR/reference_vector_generator_node.py"

# conda's bundled python is incompatible with a foreign PYTHONPATH
# (ROS setup sets one), which breaks conda itself. We rebuild it below.
unset PYTHONPATH

# --- Workspace root: walk up until we find devel/setup.bash -----------------
find_workspace_root() {
  local dir="$SCRIPT_DIR"
  while [[ "$dir" != "/" ]]; do
    if [[ -f "$dir/devel/setup.bash" ]]; then
      echo "$dir"
      return 0
    fi
    dir="$(dirname "$dir")"
  done
  return 1
}

WS_ROOT="${CATKIN_WS:-$(find_workspace_root || true)}"
if [[ -z "$WS_ROOT" ]]; then
  echo "Could not locate the catkin workspace (no devel/setup.bash found above $SCRIPT_DIR)." >&2
  echo "Set CATKIN_WS to point at it, e.g.: export CATKIN_WS=/auv_ws" >&2
  exit 1
fi

# --- ROS setup ---------------------------------------------------------------
ROS_DISTRO="${ROS_DISTRO:-}"
if [[ -z "$ROS_DISTRO" ]]; then
  for d in /opt/ros/*/setup.bash; do
    [[ -e "$d" ]] && ROS_DISTRO="$(basename "$(dirname "$d")")" && break
  done
fi
if [[ -z "$ROS_DISTRO" ]]; then
  echo "Could not determine ROS_DISTRO (no /opt/ros/* setup.bash found)." >&2
  exit 1
fi

ROS_SETUP="/opt/ros/$ROS_DISTRO/setup.bash"
if [[ ! -f "$ROS_SETUP" ]]; then
  echo "Could not find ROS setup file: $ROS_SETUP" >&2
  exit 1
fi

# --- Conda setup (search order: override, conda on PATH, common locations) --
CONDA_SH="${DINO_CONDA_SH:-}"
if [[ -z "$CONDA_SH" ]]; then
  if [[ -n "${CONDA_EXE:-}" ]] || command -v conda >/dev/null 2>&1; then
    CONDA_BIN="${CONDA_EXE:-$(command -v conda)}"
    CONDA_SH="$("$CONDA_BIN" info --base 2>/dev/null)/etc/profile.d/conda.sh"
  fi
fi
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  for d in "${HOME:-/root}/miniconda3" "${HOME:-/root}/anaconda3" /opt/conda; do
    if [[ -f "$d/etc/profile.d/conda.sh" ]]; then
      CONDA_SH="$d/etc/profile.d/conda.sh"
      break
    fi
  done
fi
if [[ ! -f "$CONDA_SH" ]]; then
  echo "Could not find conda. Install it with scripts/setup_dino_conda_env.sh" >&2
  echo "or point DINO_CONDA_SH at your conda.sh, e.g.: export DINO_CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

CONDA_ENV="${DINO_CONDA_ENV:-auto_label}"
CONDA_ROOT="$(dirname "$(dirname "$(dirname "$CONDA_SH")")")"
CONDA_PYTHON="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"
if [[ ! -x "$CONDA_PYTHON" ]]; then
  echo "Could not find conda python: $CONDA_PYTHON" >&2
  echo "Create the environment first: scripts/setup_dino_conda_env.sh" >&2
  exit 1
fi

# The conda env runs Python 3.10; give it access to the ROS (3.8) python
# modules (rospy, rospkg, genpy, generated messages) via PYTHONPATH.
ROS_PYTHON_DIR="/opt/ros/$ROS_DISTRO/lib/python3/dist-packages"
DEVEL_PYTHON_DIR="$WS_ROOT/devel/lib/python3/dist-packages"
export PYTHONPATH="$ROS_PYTHON_DIR:$DEVEL_PYTHON_DIR:${PYTHONPATH:-}"

source "$CONDA_SH"
conda activate "$CONDA_ENV"
set +u
source "$ROS_SETUP"
. "$WS_ROOT/devel/setup.bash"
set -u

pids=()

cleanup() {
  trap - INT TERM EXIT

  if ((${#pids[@]} > 0)); then
    echo
    echo "Stopping vision nodes..."
    for pid in "${pids[@]}"; do
      kill "$pid" 2>/dev/null || true
    done
    wait "${pids[@]}" 2>/dev/null || true
  fi
}

trap cleanup INT TERM EXIT

"$CONDA_PYTHON" "$DINO_NODE" &
pids+=("$!")

"$CONDA_PYTHON" "$REFERENCE_NODE" &
pids+=("$!")

echo "Started dino_tracker_node.py and reference_vector_generator_node.py in conda env: $CONDA_ENV"
echo "Press Ctrl-C to stop both nodes."

while true; do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      echo "A node exited, stopping the other node."
      cleanup
      exit "$status"
    fi
  done
  sleep 1
done
