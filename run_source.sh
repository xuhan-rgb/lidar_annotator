#!/usr/bin/env bash
# Source deployment: use a virtualenv, never unpack or seed the offline bundle.
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${LIDAR_PYTHON:-${XDG_CACHE_HOME:-$HOME/.cache}/lidar_annotator/venvs/default/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo 'Python environment missing. Run: bash scripts/setup_env.sh' >&2; exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${XDG_CACHE_HOME:-$HOME/.cache}/lidar_annotator/pycache"
unset PYTHONHOME
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools/autolabel"
DEPENDENCIES="$("$PYTHON_BIN" "$ROOT_DIR/scripts/dependencies.py" path)"
if ! "$PYTHON_BIN" "$ROOT_DIR/scripts/dependencies.py" check --quiet; then
  echo 'Remote dependencies are missing or modified. Run: python scripts/dependencies.py fetch using your application environment.' >&2
  exit 1
fi
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/tools/autolabel:$DEPENDENCIES/sam3:$DEPENDENCIES/wilddet3d:$DEPENDENCIES/wilddet3d/third_party/lingbot_depth:$DEPENDENCIES/wilddet3d/third_party/utils3d:$DEPENDENCIES/vis4d:$DEPENDENCIES/msgpack"
export LIDAR_ANNOTATOR_PATHS="${LIDAR_ANNOTATOR_PATHS:-$ROOT_DIR/configs/paths.source.json}"
export LIDAR_RUNTIME_PYTHON="$PYTHON_BIN"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/lidar_annotator/ultralytics}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
case "${1:-}" in
  models) shift; exec "$PYTHON_BIN" "$ROOT_DIR/scripts/models.py" "$@";;
  deployment-check) shift; exec "$PYTHON_BIN" "$ROOT_DIR/scripts/check_deployment.py" "$@";;
esac
exec "$PYTHON_BIN" "$ROOT_DIR/lidar_annotator.py" "$@"
