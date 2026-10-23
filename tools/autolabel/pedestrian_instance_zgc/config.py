"""Frozen paths and thresholds for SAM3/WildDet3D pedestrian instances."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import DATA,MODEL_CACHE,PROJECT_ROOT
from resource_paths import dependency_root
import hashlib
import json
import random


SOURCE=DATA/'labels/zgc'
OUT=DATA/'pedestrian_instance_labels_20261014'
YOLO=DATA/'pedestrian_xy_yolo_20261013/labels_yolo26x'
OLD_SUPERVISION=DATA/'pedestrian_xy_yolo_20261013/pedestrian_supervision'
WILDDET_TOOL=PROJECT_ROOT/'tools/autolabel/wilddet3d_zgc'
WILDDET_VENDOR=dependency_root()/'wilddet3d'
WILDDET_DEPS=WILDDET_VENDOR/'third_party'
WILDDET_WEIGHTS=MODEL_CACHE/'wilddet3d.pt'
WILDDET_WEIGHT_MANIFEST=WILDDET_WEIGHTS.parent/'download_manifest.json'
SAM3_SOURCE=dependency_root()/'sam3'
SAM3_WEIGHTS=MODEL_CACHE/'sam3.pt'
SAM3_WEIGHT_SHA256='9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e'

WILDDET_PROMPT='person'  # official WildDet3D all-prompt README example
WILDDET_COMBINED_SCORE_MIN=.6
WILDDET_SCORE_3D_MIN=.1
SAM3_PROMPT='person'
SAM3_INSTANCE_SCORE_MIN=.25
SAM3_MASK_THRESHOLD=.5
SAM3_PERSON_INDEPENDENT_CONFIRMATION_SCORE_MIN=.7
YOLO_PERSON_SCORE_MIN=.4
MIN_SUPPORT_POINTS=5
DBSCAN_MIN_SAMPLES=3
DBSCAN_EPS_MIN_M=.25
DBSCAN_EPS_MAX_M=.6
DBSCAN_EPS_FORMULA='clip(0.15+0.02*median_camera_depth,0.25,0.6)m'
MAX_XY_SPAN_M=1.4
TRAINING_ROI=(-20.,-20.,-3.,20.,20.,6.)
OLD_NEGATIVE_REL='pedestrian_xy_yolo_20261013/pedestrian_supervision/frames/{frame}.npz'
NEGATIVE_GRID_SHAPE=(80,80)
NEGATIVE_GRID_RESOLUTION_M=.5


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def choose_smoke_frames(seed=20261014,random_positive_count=2):
    """Use fixed diagnostic frames plus reproducible random YOLO-positive frames."""
    rows=json.loads((SOURCE/'index.json').read_text())['rows']
    fixed=[217,324,372]
    fixed_names={f'{f:06d}' for f in fixed}
    positives=[]
    for i,row in enumerate(rows):
        frame=Path(row['path']).stem
        if frame in fixed_names:continue
        metadata=json.loads((YOLO/'metadata'/f'{frame}.json').read_text())
        # This dedicated cache contains only YOLO26x person detections; its
        # records have no class/name field (class identity is in policy.json).
        if any(float(d.get('score',0))>=YOLO_PERSON_SCORE_MIN
               for d in metadata.get('detections',[])):
            positives.append(i)
    selected=random.Random(seed).sample(positives,min(random_positive_count,len(positives)))
    return dict(seed=seed,selection='uniform random sample without replacement from all YOLO-positive frames, excluding fixed diagnostics',
        positive_frame_population=len(positives),frames=sorted(fixed+selected))
