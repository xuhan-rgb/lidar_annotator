"""Project-local path configuration; no imports from training projects."""
import json,os
from pathlib import Path
PROJECT_ROOT=Path(__file__).resolve().parents[2]
CONFIG=Path(os.environ.get('LIDAR_ANNOTATOR_PATHS',PROJECT_ROOT/'configs/paths.source.json'))
PATHS=json.loads(CONFIG.read_text())
def configured(name):
    p=Path(PATHS[name]).expanduser()
    return p if p.is_absolute() else (CONFIG.parent/p).resolve()
DATA=configured('data_root')
MODEL_CACHE=Path(os.environ.get('LIDAR_MODEL_CACHE',configured('model_cache'))).expanduser()
