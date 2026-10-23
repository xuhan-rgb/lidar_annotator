"""Fixed production recipe for the confirmed WildDet3D + SAM instance experiment."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import DATA,MODEL_CACHE,PROJECT_ROOT
from resource_paths import dependency_root


SOURCE = DATA / "labels/zgc"
RAW_ROOT = DATA / "wilddet3d_zgc_singleframe_20261014"
DEFAULT_OUTPUT_ROOT = DATA / "wilddet3d_instance_labels_20261014"
CONFIRMED_FIVE_FRAME_ROOT = DATA / "wilddet3d_instance_association_20261014"
FRAME_IDS = (300, 324, 372, 503, 823)
PROMPTS = ("car", "van", "truck")
INSTANCE_CANDIDATE_THRESHOLD = 0.25
MASK_PIXEL_THRESHOLD = 0.5
DISPLAY_SCORE_THRESHOLD = 0.6
DISPLAY_SCORE_3D_THRESHOLD = 0.1
MIN_SELECTED_POINTS = 10
DBSCAN_EPS_METERS = 0.4
DBSCAN_MIN_SAMPLES = 3
TRAINING_ROI_XY = (-20.0, 20.0)
TRAINING_ROI_Z = (-3.0, 6.0)
CHECKPOINT_PATH = MODEL_CACHE/'sam3.pt'
CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_SOURCE = dependency_root()/'sam3'
PROMPT_DEDUP_MASK_IOU = 0.9
PROMPT_DEDUP_BOX_IOU = 0.9
ASSOC_BOX_IOU_MIN = 0.1
ASSOC_MASK_COVERAGE_MIN = 0.25
ASSOC_UTILITY_MARGIN_MIN = 0.05
HUBER_DELTA_METERS = 0.15
OBJECTIVE_SURFACE_WEIGHT = 1.0
OBJECTIVE_FIRST_RAY_WEIGHT = 1.0
