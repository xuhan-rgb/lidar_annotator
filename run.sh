#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo 'This offline runtime supports Linux x86_64 only.' >&2; exit 1
fi
SOURCE_PYTHON="${LIDAR_PYTHON:-${XDG_CACHE_HOME:-$HOME/.cache}/lidar_annotator/venvs/default/bin/python}"
if [[ -x "$SOURCE_PYTHON" ]]; then
  export LIDAR_PYTHON="$SOURCE_PYTHON"
  exec bash "$ROOT_DIR/run_source.sh" "$@"
fi
RUNTIME_ID="$(cat "$ROOT_DIR/configs/runtime-id")"
BUNDLE_DIR="${LIDAR_OFFLINE_BUNDLE:-$HOME/.cache/lidar_annotator/offline-bundles/$RUNTIME_ID}"
ARCHIVE="$BUNDLE_DIR/python-linux-x86_64.tar"
CACHE_ROOT="${LIDAR_RUNTIME_CACHE:-$HOME/.cache/lidar_annotator/runtime}"
RUNTIME_DIR="$CACHE_ROOT/$RUNTIME_ID"
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME:-$HOME/.cache}/lidar_annotator/pycache"
unset PYTHONHOME PYTHONPATH
export LIDAR_ANNOTATOR_PATHS="$ROOT_DIR/configs/paths.source.json"
export QT_PLUGIN_PATH="$RUNTIME_DIR/lib/python3.12/site-packages/PySide6/Qt/plugins"
export LIDAR_RUNTIME_PYTHON="$RUNTIME_DIR/bin/python"
export LD_LIBRARY_PATH="$BUNDLE_DIR/system-libs:$RUNTIME_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$CACHE_ROOT"
(
  flock 9
  if [[ ! -f "$RUNTIME_DIR/.install-prefix" ]] || [[ "$(cat "$RUNTIME_DIR/.install-prefix")" != "$RUNTIME_DIR" ]]; then
    if [[ ! -f "$ARCHIVE" ]]; then echo 'Cached offline runtime archive is missing. Use bash scripts/setup_env.sh and bash run_source.sh.' >&2; exit 1; fi
    echo 'Preparing the bundled Python / Qt / CUDA runtime. No network download is required.'
    rm -rf -- "$RUNTIME_DIR"
    mkdir -p "$RUNTIME_DIR"
    tar -xf "$ARCHIVE" -C "$RUNTIME_DIR"
    "$RUNTIME_DIR/bin/python" "$RUNTIME_DIR/bin/conda-unpack"
    printf '%s\n' "$RUNTIME_DIR" > "$RUNTIME_DIR/.install-prefix"
  fi
) 9> "$CACHE_ROOT/.install.lock"
export LIDAR_PYTHON="$RUNTIME_DIR/bin/python"
exec bash "$ROOT_DIR/run_source.sh" "$@"
