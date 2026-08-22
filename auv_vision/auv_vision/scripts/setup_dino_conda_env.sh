#!/usr/bin/env bash
set -euo pipefail

# Bootstraps a conda env with Python 3.10 for the DINO nodes.
# The ROS nodes themselves keep running on the system Python 3.8.
# Everything is resolved from the environment — no hardcoded paths.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${DINO_CONDA_ENV:-auto_label}"
MINICONDA_DIR="${DINO_MINICONDA_DIR:-${HOME:-/root}/miniconda3}"
MINICONDA_URL="${DINO_MINICONDA_URL:-https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh}"
TORCH_INDEX_URL="${DINO_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"

# conda's bundled python is incompatible with a foreign PYTHONPATH
# (ROS setup sets one), which breaks conda itself.
unset PYTHONPATH

# --- Locate or install conda ------------------------------------------------
CONDA_SH=""
if [[ -n "${CONDA_EXE:-}" ]] || command -v conda >/dev/null 2>&1; then
  CONDA_BIN="${CONDA_EXE:-$(command -v conda)}"
  CONDA_SH="$("$CONDA_BIN" info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  if [[ -f "$MINICONDA_DIR/etc/profile.d/conda.sh" ]]; then
    CONDA_SH="$MINICONDA_DIR/etc/profile.d/conda.sh"
  fi
fi

if [[ -z "$CONDA_SH" || ! -f "$CONDA_SH" ]]; then
  echo "conda not found. Installing Miniconda to: $MINICONDA_DIR"
  curl -fsSL "$MINICONDA_URL" -o /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "$MINICONDA_DIR"
  rm -f /tmp/miniconda.sh
  CONDA_SH="$MINICONDA_DIR/etc/profile.d/conda.sh"
fi

source "$CONDA_SH"
conda config --set auto_activate_base false

# --- Create / update the environment ----------------------------------------
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "Creating conda environment '$CONDA_ENV' with Python 3.10..."
  # conda-forge avoids Anaconda's channel Terms-of-Service prompt.
  conda create -y -n "$CONDA_ENV" --override-channels -c conda-forge python=3.10 pip
else
  echo "Conda environment '$CONDA_ENV' already exists."
fi

conda activate "$CONDA_ENV"

# --- Install packages ---------------------------------------------------------
# dinov3 (requirements.txt) + ROS python bindings for the conda interpreter.
python -m pip install --upgrade pip
python -m pip install \
  numpy opencv-python-headless pillow rospkg catkin_pkg \
  ftfy omegaconf regex scikit-learn submitit termcolor torchmetrics
python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"

echo
echo "Done. Environment '$CONDA_ENV' is ready at:"
echo "  $(command -v python)"
echo
echo "Start the nodes with:"
echo "  roslaunch auv_vision dino.launch"
echo "or run the launcher directly:"
echo "  $SCRIPT_DIR/start_dino_nodes.sh"
