#!/usr/bin/env bash
# Install the source distribution without the bundled offline runtime.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
unset PYTHONHOME PYTHONPATH
VENV_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/lidar_annotator/venvs/default"
PYTHON_BIN=python3.12
TORCH_INDEX=https://download.pytorch.org/whl/cu128
while (($#)); do
  case "$1" in
    --python|--venv|--torch-index)
      if (($# < 2)); then echo "Missing value: $1" >&2; exit 2; fi
      case "$1" in
        --python) PYTHON_BIN="$2";;
        --venv) VENV_DIR="$2";;
        --torch-index) TORCH_INDEX="$2";;
      esac
      shift 2;;
    -h|--help)
      echo 'Usage: bash scripts/setup_env.sh [--python python3.12] [--venv PATH] [--torch-index URL]'
      echo 'Installs dependencies only. Download weights with scripts/models.py download.'
      exit 0;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'Full inference currently targets Linux x86_64 with an NVIDIA GPU.' >&2; exit 1
fi
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12"'
VENV_DIR="$("$PYTHON_BIN" -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$VENV_DIR")"
"$PYTHON_BIN" -m venv "$VENV_DIR"
PYTHON_ENV="$VENV_DIR/bin/python"
"$PYTHON_ENV" "$ROOT_DIR/scripts/dependencies.py" fetch
"$PYTHON_ENV" -m pip install --upgrade pip
"$PYTHON_ENV" -m pip install torch==2.10.0 torchvision==0.25.0 --index-url "$TORCH_INDEX"
"$PYTHON_ENV" -m pip install -r "$ROOT_DIR/requirements/inference.txt"
"$PYTHON_ENV" -m pip check
LIDAR_PYTHON="$PYTHON_ENV" bash "$ROOT_DIR/run_source.sh" deployment-check
echo 'Environment ready. Next: bash run_source.sh models download (set LIDAR_PYTHON if you chose a custom environment).'
