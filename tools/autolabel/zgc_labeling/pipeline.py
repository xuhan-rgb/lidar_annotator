#!/usr/bin/env python3
"""Run the explicit ZGC semantic, instance, and temporal stages."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import DATA,PROJECT_ROOT


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/zgc_backend.json"
SEMANTIC_RUNNER = PROJECT_ROOT / "tools/autolabel/semantic_zgc/run_pipeline.py"
PEDESTRIAN_DIR = REPO_ROOT / "tools/autolabel/pedestrian_instance_zgc"
VEHICLE_DIR = REPO_ROOT / "tools/autolabel/wilddet3d_zgc"
DATA_ROOT = DATA
YOLO_POLICY = DATA_ROOT / "pedestrian_xy_yolo_20261013/labels_yolo26x/policy.json"
SAM_CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
YOLO_DETECTOR_SHA256 = "9fdd44a31c504547ffb81d2c6d9e6dac3493c8eaa8b0398d3f43bae6c7003e92"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path):
    return json.loads(Path(path).read_text())


def _run(command, cwd, runner):
    result = runner([sys.executable, *map(str, command)], cwd=str(cwd), check=True)
    return getattr(result, "returncode", 0)


def _validate_semantic_cache(source_index):
    source_index = Path(source_index)
    if not source_index.is_file():
        raise RuntimeError(f"Semantic source index missing: {source_index}")
    index = _load(source_index)
    rows = index.get("rows", [])
    if len(rows) != 971:
        raise RuntimeError(f"Semantic cache must contain 971 indexed frames; found {len(rows)}")
    ids = [Path(row["path"]).stem for row in rows]
    expected = [f"{i:06d}" for i in range(971)]
    if sorted(ids) != expected:
        raise RuntimeError("Semantic source index has missing, duplicate, or unexpected frame IDs")

    verification = _load(source_index.parent.parent / "verification.json")
    if verification.get("state") != "complete" or verification.get("frames") != 971:
        raise RuntimeError("Semantic verification report is not complete for all 971 frames")

    for row in rows:
        frame = Path(row["path"]).stem
        output = source_index.parent / row["path"]
        if not output.is_file() or _sha256(output) != row.get("sha256"):
            raise RuntimeError(f"Semantic output missing or SHA mismatch for frame {frame}")
        source_path = row.get("source_label_path")
        source_sha = row.get("source_label_sha256")
        if not source_path or not source_sha or not Path(source_path).is_file():
            raise RuntimeError(f"Semantic input provenance missing for frame {frame}")
        if _sha256(source_path) != source_sha:
            raise RuntimeError(f"Semantic source-label SHA mismatch for frame {frame}")
        original_sha = row.get("original_payload_sha256", "")
        if len(original_sha) != 64:
            raise RuntimeError(f"Original source payload SHA missing for frame {frame}")
        for folder, suffix in (("image_masks", ".png"), ("confidence", ".npz"),
                               ("metadata", ".json"), ("overlay", ".jpg")):
            if not (source_index.parent / folder / f"{frame}{suffix}").is_file():
                raise RuntimeError(f"Semantic cache incomplete: {folder}/{frame}{suffix} missing")
    return {"status": "cached", "frames": 971, "source_index_sha256": _sha256(source_index)}


def _validate_person_cache(source_index, person_root):
    person_root = Path(person_root)
    source_hash = _sha256(source_index)
    source_rows = _load(source_index).get("rows", [])
    source_ids = {Path(row["path"]).stem: row.get("frame_id") for row in source_rows}
    index = _load(person_root / "index.json")
    if index.get("source_index_sha256") != source_hash:
        raise RuntimeError("Pedestrian cache was generated from a different semantic source index")
    entries = index.get("frames", [])
    if len(entries) != 971 or {item.get("frame") for item in entries} != {f"{i:06d}" for i in range(971)}:
        raise RuntimeError("Pedestrian cache must index exactly the current 971 frames")
    for item in entries:
        frame = item["frame"]
        npz = person_root / item["npz"]
        metadata = person_root / item["metadata"]
        if not npz.is_file() or _sha256(npz) != item.get("npz_sha256"):
            raise RuntimeError(f"Pedestrian NPZ absent or SHA mismatch for frame {frame}")
        if not metadata.is_file() or _sha256(metadata) != item.get("metadata_sha256"):
            raise RuntimeError(f"Pedestrian metadata absent or SHA mismatch for frame {frame}")

    report = _load(person_root / "yolo_only_regeneration_report.json")
    if report.get("status") != "passed" or report.get("frames") != 971:
        raise RuntimeError("Pedestrian YOLO-only regeneration report is missing or incomplete")
    if report.get("current_pedestrian_xy") != 1373 or report.get("manual_ground_truth") is not False:
        raise RuntimeError("Pedestrian cache does not match the confirmed current production policy")
    code_hashes = report.get("code_sha256", {})
    for name in ("process_instances.py", "validate_all.py"):
        current = _sha256(PEDESTRIAN_DIR / name)
        if code_hashes.get(name) != current:
            raise RuntimeError(f"Pedestrian cache code signature changed: {name}")
    policy = _load(YOLO_POLICY)
    if (policy.get("class_id") != 0 or policy.get("yolo_confidence") != 0.4
            or policy.get("image_size") != 1280
            or policy.get("detector_sha256") != YOLO_DETECTOR_SHA256):
        raise RuntimeError("Cached YOLO26x person policy differs from conf=0.4/imgsz=1280")
    yolo_root = YOLO_POLICY.parent
    yolo_status = _load(yolo_root / "status.json")
    yolo_index = _load(yolo_root / "index.json")
    if yolo_status.get("stage") != "complete" or yolo_status.get("frames") != 971:
        raise RuntimeError("YOLO26x person detection cache is incomplete")
    yolo_rows = yolo_index.get("rows", [])
    if len(yolo_rows) != 971 or {row.get("frame") for row in yolo_rows} != {f"{i:06d}" for i in range(971)}:
        raise RuntimeError("YOLO26x person cache index does not cover the current 971 frames")
    for row in yolo_rows:
        frame = row["frame"]
        detections = _load(yolo_root / "metadata" / f"{frame}.json")
        if row.get("frame_id") != source_ids.get(frame) or detections.get("frame_id") != row.get("frame_id"):
            raise RuntimeError(f"YOLO26x detection source ID mismatch for frame {frame}")
    sam_report = _load(person_root / "raw_sam_person/run_report.json")
    if (sam_report.get("stage") != "complete" or sam_report.get("frames") != 971
            or sam_report.get("instance_candidate_threshold") != 0.25
            or sam_report.get("mask_pixel_threshold") != 0.5
            or sam_report.get("checkpoint_sha256") != SAM_CHECKPOINT_SHA256):
        raise RuntimeError("SAM person instance cache does not match the confirmed inference policy")
    sam_root = person_root / "raw_sam_person"
    for frame in range(971):
        stem = f"{frame:06d}"
        sam_meta = _load(sam_root / "metadata" / f"{stem}.json")
        mask_path = sam_root / "frames" / f"{stem}.npz"
        if (sam_meta.get("frame_id") != source_ids.get(stem)
                or not mask_path.is_file() or _sha256(mask_path) != sam_meta.get("mask_sha256")):
            raise RuntimeError(f"SAM person mask cache absent or invalid for frame {stem}")
    return {"status": "cached", "frames": 971, "labels": 1373,
            "source_index_sha256": source_hash, "confirmation_policy": "YOLO26x person, conf=0.4, imgsz=1280"}


def _validate_vehicle_cache(source_index, vehicle_root):
    source_index = Path(source_index)
    vehicle_root = Path(vehicle_root)
    rows = _load(source_index).get("rows", [])
    by_frame = {int(Path(row["path"]).stem): row for row in rows}
    if len(by_frame) != 971:
        raise RuntimeError("Vehicle source index must contain all 971 frames")

    sys.path.insert(0, str(VEHICLE_DIR))
    try:
        from generate_vehicle_camera_ray_distance030_batch import _algorithm_signature, _resume_valid
        signature = _algorithm_signature()
        recorded = _load(vehicle_root / "reports/generation_signature.json")
        # Existing cached labels retain the original production signature. Only a
        # recorded, explicitly bundled migration signature may be reused.
        provenance=_load(PROJECT_ROOT/'provenance.json')
        original={Path(v['file']).name:v['original_sha256'] for v in provenance['files'] if 'wilddet3d_zgc' in v['file']}
        names=('process_vehicle_camera_ray_distance.py','center_distance_selection.py','review_bev_pose.py','associate_instance_masks.py','verify_vehicle_class_000402.py','refine_instance_support.py','instance_label_config.py','label_geometry.py')
        old_policy=dict(score_threshold=.3,score_3d_threshold=.1,code_sha256={name:original[name] for name in names})
        migrated=hashlib.sha256(json.dumps(old_policy,sort_keys=True).encode()).hexdigest()
        if recorded.get("signature") not in (signature,migrated):
            raise RuntimeError("Unknown vehicle cache generation signature")
        signature=recorded['signature']
        summary = _load(vehicle_root / "summary.json")
        if summary.get("status") != "complete" or summary.get("frame_count") != 971:
            raise RuntimeError("Vehicle cache does not have a complete 971-frame summary")
        for frame, row in by_frame.items():
            if not _resume_valid(vehicle_root, frame, row.get("original_payload_sha256"),expected_signature=signature):
                raise RuntimeError(f"Vehicle cache signature/source validation failed for frame {frame:06d}")
    finally:
        sys.path.pop(0)
    return {"status": "cached", "frames": 971, "labels": summary.get("accepted_label_count"),
            "generation_signature": signature}


def _load_config(path):
    config = _load(path)
    required = ("source_index", "person_root", "vehicle_root", "pose_npz", "pose_report",
                "odometry_npz", "temporal_output")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")
    return config


def run_stages(config_path=DEFAULT_CONFIG, refresh_semantic=False, refresh_instances=False,
               workers=8, frames=None, runner=subprocess.run, temporal_script=None):
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    report = {"started_unix": time.time(), "manual_ground_truth": False,
              "added_to_training": False, "stages": {}}
    source_index = Path(config["source_index"])
    if refresh_semantic:
        _run([SEMANTIC_RUNNER], SEMANTIC_RUNNER.parent, runner)
        report["stages"]["semantic"] = {"status": "refreshed"}
    report["stages"]["semantic_cache"] = _validate_semantic_cache(source_index)

    if refresh_instances:
        _run(["process_instances.py", "--all"], PEDESTRIAN_DIR, runner)
        _run(["generate_vehicle_camera_ray_distance030_batch.py", "--workers", workers], VEHICLE_DIR, runner)
        report["stages"]["instances"] = {"status": "refreshed", "workers": workers}
    report["stages"]["pedestrian_cache"] = _validate_person_cache(source_index, config["person_root"])
    report["stages"]["vehicle_cache"] = _validate_vehicle_cache(source_index, config["vehicle_root"])

    temporal_script = Path(temporal_script or HERE / "temporal_refine.py")
    temporal_args = [temporal_script, "--config", config_path]
    if frames is not None:
        temporal_args.extend(["--frames", *frames])
    _run(temporal_args, HERE, runner)
    report["stages"]["temporal"] = {"status": "dispatched", "output": config["temporal_output"],
                                     "frames": frames or "all"}
    report["status"] = "dispatched"
    report_path = DATA_ROOT / "zgc_pipeline_20261014" / "run_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    return report


def run_temporal(config_path=DEFAULT_CONFIG, frames=None, runner=subprocess.run, temporal_script=None):
    config_path = Path(config_path).resolve()
    config = _load_config(config_path)
    temporal_script = Path(temporal_script or HERE / "temporal_refine.py")
    command = [temporal_script, "--config", config_path]
    if frames is not None:
        command.extend(["--frames", *frames])
    return _run(command, HERE, runner)


def main(argv=None, runner=subprocess.run):
    parser = argparse.ArgumentParser(description="Three-stage ZGC labeling and temporal-refinement wrapper")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Validate/reuse semantic and instance caches, then run temporal refinement")
    run.add_argument("--config", default=str(DEFAULT_CONFIG))
    run.add_argument("--refresh-semantic", action="store_true", help="Explicitly invoke DATA/run_pipeline.py")
    run.add_argument("--refresh-instances", action="store_true", help="Explicitly regenerate current person and vehicle labels")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--frames", type=int, nargs="+", help="Optional frame subset for temporal refinement only")
    temporal = commands.add_parser("temporal", help="Run only the temporal stage")
    temporal.add_argument("--config", default=str(DEFAULT_CONFIG))
    temporal.add_argument("--frames", type=int, nargs="+")
    args = parser.parse_args(argv)
    if args.command == "run":
        report = run_stages(args.config, args.refresh_semantic, args.refresh_instances,
                            args.workers, args.frames, runner=runner)
        print(json.dumps(report, indent=2))
        return 0
    return run_temporal(args.config, args.frames, runner=runner)


if __name__ == "__main__":
    raise SystemExit(main())
