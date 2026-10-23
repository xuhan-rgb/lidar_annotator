"""Locations of commit-locked remote sources and fonts in the user cache."""
import hashlib
import os
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
LOCK_PATH=PROJECT_ROOT/'configs/dependencies.lock.json'


def dependency_root(lock_path=LOCK_PATH,cache_base=None):
    identity=hashlib.sha256(Path(lock_path).read_bytes()).hexdigest()[:20]
    cache=Path(cache_base) if cache_base is not None else Path(os.environ.get('XDG_CACHE_HOME',Path.home()/'.cache'))/'lidar_annotator/dependencies'
    return cache.expanduser().resolve()/identity


def font_path(name):
    if name not in ('DejaVuSans.ttf','NotoSansCJKsc-Regular.otf'):raise ValueError(name)
    return dependency_root()/'fonts'/name


def python_paths(root=None):
    root=Path(root) if root is not None else dependency_root()
    return [root/'sam3',root/'wilddet3d',root/'wilddet3d/third_party/lingbot_depth',
            root/'wilddet3d/third_party/utils3d',root/'vis4d',root/'msgpack']
