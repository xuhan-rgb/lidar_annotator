#!/usr/bin/env python3
"""Generate fresh YOLO26x/WildDet3D-confirmed labels from a semantic job cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from task_progress import report_progress
from runtime_paths import MODEL_CACHE  # noqa: E402
from resource_paths import dependency_root


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()



def _link_processing_input(source, destination):
    """Expose immutable model inputs without storing another byte copy."""
    source = Path(source); destination = Path(destination)
    if not source.is_file(): raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    try:
        temporary.symlink_to(os.path.relpath(source.resolve(), destination.parent.resolve()))
        os.replace(temporary, destination)
    finally:
        if temporary.is_symlink(): temporary.unlink()


def _source_aliases(source_path, destination):
    with np.load(source_path) as archive:
        data = {key: archive[key] for key in archive.files}
    complete = all(key in data for key in ("cam_intrinsic", "cam_extrinsic", "lidar_timestamp_ns", "point_labels"))
    data["cam_intrinsic"] = data.get("cam_intrinsic", data.get("K"))
    data["cam_extrinsic"] = data.get("cam_extrinsic", data.get("E"))
    data["lidar_timestamp_ns"] = data.get("lidar_timestamp_ns", data.get("timestamp"))
    if any(data.get(key) is None for key in ("cam_intrinsic", "cam_extrinsic", "lidar_timestamp_ns")):
        raise ValueError(f"Source frame lacks K/E/timestamp fields: {source_path}")
    # The legacy instance consumer only uses this field for a diagnostic count.
    data.setdefault("point_labels", np.zeros(len(data["xyz"]), dtype=np.uint8))
    if complete:
        _link_processing_input(source_path, destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(temporary, **data)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _load_instances(path, metadata_path=None):
    with np.load(path) as z:
        shape = tuple(map(int, z["mask_shape"].tolist()))
        packed = z["masks_packed"].copy()
    masks = np.unpackbits(packed, axis=1, count=int(np.prod(shape)), bitorder="little").reshape(-1, *shape).astype(bool)
    records = json.loads(Path(metadata_path or path.with_suffix(".json")).read_text())["records"]
    if len(records) != len(masks):
        raise ValueError(f"SAM record/mask count mismatch: {path}")
    return masks, records


def _prepare(job, rows):
    data = job / "data"
    source = data / "source"
    yolo_root = data / "yolo"
    sam_person = data / "raw_sam_person"
    sam_vehicle = data / "vehicle_masks"
    wild_root = data / "raw_wilddet"
    for root in (source, yolo_root, sam_person, sam_vehicle, wild_root):
        root.mkdir(parents=True, exist_ok=True)
    source_rows = []
    for row in rows:
        record = row.get("source_index_record", {})
        if record.get("min_gt_obstacle_distance_m") == float("inf"):
            # Export the source's unavailable-distance sentinel as finite JSON.
            row = dict(row, source_index_record=dict(record, min_gt_obstacle_distance_m=1e6))
        source_rows.append(row)
    _json(source / "index.json", {"rows": source_rows})
    return source, yolo_root, sam_person, sam_vehicle, wild_root


def _run_models(job, rows, source, yolo_root, sam_person, sam_vehicle, wild_root, refresh=False):
    import torch
    from ultralytics import YOLO

    yolo_weights = MODEL_CACHE / "yolo26x.pt"
    wild_weights = MODEL_CACHE / "wilddet3d.pt"
    if not yolo_weights.is_file() or not wild_weights.is_file():
        raise FileNotFoundError(f"Required model weights missing under {MODEL_CACHE}")
    model_manifest = json.loads((PROJECT_ROOT / "models" / "manifest.json").read_text())
    yolo_hash, wild_hash = _sha256(yolo_weights), _sha256(wild_weights)
    for name, path, digest in (("yolo26x.pt", yolo_weights, yolo_hash),
                               ("wilddet3d.pt", wild_weights, wild_hash)):
        expected = model_manifest[name]
        if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
            raise ValueError(f"Model weight checksum/size mismatch: {path}")
    wild_download = json.loads((MODEL_CACHE / "download_manifest.json").read_text())
    if not wild_download.get("official_lfs_sha_verified") or wild_hash != wild_download.get("sha256"):
        raise ValueError("WildDet3D checkpoint is not the verified official LFS checkpoint")
    dependencies = dependency_root()
    wild_vendor = dependencies / "wilddet3d"
    for p in (wild_vendor, wild_vendor / "third_party/utils3d",
              wild_vendor / "third_party/lingbot_depth", dependencies / "vis4d"):
        sys.path.insert(0, str(p))
    import termcolor  # noqa: F401
    from wilddet3d import build_model, preprocess
    from vis4d.op.box.box3d import boxes3d_to_corners
    from vis4d.data.const import AxisMode

    config_path = PROJECT_ROOT / "models" / "lingbot_model_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Bundled LingBot model config missing: {config_path}")
    os.environ["WILDDET3D_LINGBOT_CONFIG"] = str(config_path)
    config_hash = _sha256(config_path)
    torch.set_num_threads(4)
    yolo_rows = []
    yolo_valid = {}
    for row in rows:
        stem = Path(row["path"]).stem
        cache = yolo_root / "metadata" / f"{stem}.json"
        saved = None
        if not refresh and cache.is_file():
            try:
                saved = json.loads(cache.read_text())
                yolo_valid[stem] = (not refresh and saved.get("frame_id") == row["frame_id"]
                                    and saved.get("source_payload_sha256") == row["original_payload_sha256"]
                                    and saved.get("weights_sha256") == yolo_hash
                                    and saved.get("confidence") == .4 and saved.get("image_size") == 1280)
            except (OSError, json.JSONDecodeError):
                yolo_valid[stem] = False
        else:
            yolo_valid[stem] = False
    yolo_model = YOLO(str(yolo_weights)) if not all(yolo_valid.values()) else None
    report_progress("instances",0,3*len(rows),detail="YOLO26x",job=job)
    for ordinal,row in enumerate(rows,1):
        stem = Path(row["path"]).stem
        if yolo_valid[stem]:
            detections = saved = json.loads((yolo_root / "metadata" / f"{stem}.json").read_text())["detections"]
        else:
            image = Image.open(row["image_path"]).convert("RGB")
            result = yolo_model.predict(source=image, conf=.4, imgsz=1280, verbose=False)[0]
            names = result.names
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.cpu()
                for j in range(len(boxes)):
                    cls = int(boxes.cls[j]); score = float(boxes.conf[j])
                    detections.append({"id": j, "class_id": cls, "name": str(names[cls]),
                                       "score": score, "bbox_xyxy": boxes.xyxy[j].tolist()})
            _json(yolo_root / "metadata" / f"{stem}.json", {
                "frame_id": row["frame_id"], "source_payload_sha256": row["original_payload_sha256"],
                "weights_sha256": yolo_hash, "confidence": .4, "image_size": 1280,
                "detections": detections})
        yolo_rows.append({"frame": stem, "frame_id": row["frame_id"], "detections": len(detections)})
        report_progress("instances",ordinal,3*len(rows),frame=row["frame_id"],detail="YOLO26x",job=job)
    _json(yolo_root / "index.json", {"frames": yolo_rows})
    if yolo_model is not None:
        del yolo_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    recipe=json.loads((job / "recipe.json").read_text()) if (job / "recipe.json").is_file() else None
    person_prompts=set(recipe["base_semantics"]["prompts"]["1"]) if recipe else {"person"}
    vehicle_prompts=set(recipe["base_semantics"]["prompts"]["2"]) if recipe else {"car","van","truck"}
    for row in rows:
        stem = Path(row["path"]).stem
        masks, records = _load_instances(job / "data/visual_instances" / f"{stem}.npz")
        for root, prompts in ((sam_person, person_prompts), (sam_vehicle, vehicle_prompts)):
            chosen = [j for j, rec in enumerate(records) if rec["prompt"] in prompts]
            selected = masks[chosen]
            selected_records = []
            for j in chosen:
                rec = records[j]
                selected_records.append({"source_instance_id": int(rec["source_instance_id"]),
                                         "prompt": rec["prompt"], "score": float(rec["score"]),
                                         "bbox_xyxy": rec["bbox_xyxy"], "area_pixels": int(rec["area_pixels"]),"track_id":rec.get("track_id")})
            cache_npz = root / "frames" / f"{stem}.npz"
            cache_npz.parent.mkdir(parents=True, exist_ok=True)
            shape = masks.shape[1:]
            np.savez_compressed(cache_npz, masks_packed=np.packbits(
                                    selected.reshape(len(selected), int(np.prod(shape))), axis=1, bitorder="little"),
                                mask_shape=np.asarray(shape, np.int32))
            cache_meta = {"frame_id": row["frame_id"], "candidates": selected_records,
                          "mask_sha256": _sha256(cache_npz), "image_source_sha256": row["original_payload_sha256"]}
            cache_meta["records"] = selected_records
            _json(root / "metadata" / f"{stem}.json", cache_meta)
    def cached_wild(row):
        path = wild_root / "raw_predictions" / f"{Path(row['path']).stem}.json"
        if refresh or not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text())
            if (raw.get("source_frame_id") == row["frame_id"]
                    and raw.get("source_payload_sha256") == row["original_payload_sha256"]
                    and raw.get("weight_sha256") == wild_hash
                    and raw.get("lingbot_config_sha256") == config_hash
                    and raw.get("mode") == "text_car"
                    and raw.get("score_threshold") == .1
                    and raw.get("score_3d_threshold") == .1):
                return raw
        except (OSError, json.JSONDecodeError):
            pass
        return None

    wild_cached = {Path(row["path"]).stem: cached_wild(row) for row in rows}
    model = None
    if any(value is None for value in wild_cached.values()):
        model = build_model(checkpoint=str(wild_weights), skip_pretrained=True,
                            score_threshold=.1, score_3d_threshold=.1, canonical_rotation=True)
        model.eval().cuda()
    for i, row in enumerate(rows):
        stem = Path(row["path"]).stem
        raw = wild_cached[stem]
        if raw is None:
            image = Image.open(row["image_path"]).convert("RGB")
            rgb = np.asarray(image)
            with np.load(source / row["path"]) as z:
                K, E = z["cam_intrinsic"], z["cam_extrinsic"]
            data = preprocess(rgb.astype(np.float32), K)
            args = dict(images=data["images"].cuda(), intrinsics=data["intrinsics"].cuda()[None],
                        input_hw=[data["input_hw"]], original_hw=[data["original_hw"]],
                        padding=[data["padding"]], input_texts=["car"])
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**args)
            boxes, cuboids, scores, scores2, scores3, classids, _depth = outputs
            to_np = lambda x: x[0].detach().float().cpu().numpy()
            bbox, cube = to_np(boxes), to_np(cuboids)
            score, score2, score3 = to_np(scores), to_np(scores2), to_np(scores3)
            ids = classids[0].detach().cpu().numpy()
            corners = boxes3d_to_corners(torch.tensor(cube, dtype=torch.float64), AxisMode.OPENCV).numpy()
            items = [{"index": j, "wilddet_instance_id": j, "category": "car",
                      "score": float(score[j]), "score_2d": float(score2[j]), "score_3d": float(score3[j]),
                      "bbox_xyxy": bbox[j].tolist(), "box_camera_center_wlh_quaternion": cube[j].tolist(),
                      "corners_cam": corners[j].tolist(), "class_id": int(ids[j])} for j in range(len(score))]
            raw = {"frame": stem, "source_frame_id": row["frame_id"],
                   "source_payload_sha256": row["original_payload_sha256"], "weight_sha256": wild_hash,
                   "lingbot_config_sha256": config_hash, "score_threshold": .1,
                   "score_3d_threshold": .1, "vehicle_label_threshold": .3,
                   "mode": "text_car", "camera_K": K.tolist(), "lidar_to_camera": E.tolist(), "items": items}
            _json(wild_root / "raw_predictions" / f"{stem}.json", raw)
            del args, outputs, data
        _link_processing_input(wild_root / "raw_predictions" / f"{stem}.json",
                               wild_root / "frames" / f"{stem}.json")
        report_progress("instances",len(rows)+i+1,3*len(rows),frame=row["frame_id"],detail="WildDet3D",job=job)


def _materialize_processing_inputs(job, rows, source, sam_person, sam_vehicle, yolo_root, wild_root):
    data = job / "data"
    old = data / "no_negative_supervision"
    for row in rows:
        stem = Path(row["path"]).stem
        old_path = old / "frames" / f"{stem}.npz"
        old_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(old_path, negative_mask=np.zeros((80, 80), bool),
                            xy=np.empty((0, 2), np.float32), confidence=np.empty(0, np.float32))
        # The pedestrian processor is given its person-only detections through a scoped loader.
        ymeta = json.loads((yolo_root / "metadata" / f"{stem}.json").read_text())
        persons = [dict(id=d["id"], score=d["score"], bbox_xyxy=d["bbox_xyxy"])
                   for d in ymeta["detections"] if d["name"] == "person"]
        _json(data / "person_yolo" / f"{stem}.json", {"frame_id": row["frame_id"], "detections": persons})
        _json(wild_root / "person_frames" / f"{stem}.json", {"frame_id": row["frame_id"], "items": []})
        # Vehicle consumer expects raw predictions at RAW_ROOT/raw_predictions.
        if not (wild_root / "raw_predictions" / f"{stem}.json").exists():
            _json(wild_root / "raw_predictions" / f"{stem}.json", {"source_payload_sha256": row["original_payload_sha256"], "items": []})


def _run_labelers(job, rows, source, yolo_root, sam_person, sam_vehicle, wild_root):
    ped_dir = PROJECT_ROOT / "tools/autolabel/pedestrian_instance_zgc"
    veh_dir = PROJECT_ROOT / "tools/autolabel/wilddet3d_zgc"
    sys.path.insert(0, str(ped_dir)); sys.path.insert(0, str(veh_dir))
    import config as ped_config
    import process_instances as pedestrian
    import instance_label_config as vehicle_config
    import process_vehicle_camera_ray_distance as vehicle
    import verify_vehicle_class_000402 as vehicle_verify
    vehicle_confirm_classes = vehicle.confirm_vehicle_classes

    data = job / "data"
    local_source = source
    ped_config.SOURCE = local_source
    ped_config.YOLO = yolo_root
    ped_config.OLD_SUPERVISION = data / "no_negative_supervision"
    pedestrian.SOURCE = local_source
    pedestrian.OLD_SUPERVISION = ped_config.OLD_SUPERVISION
    pedestrian.YOLO = yolo_root
    person_root, vehicle_root = data / "person", data / "vehicle"
    vehicle_config.SOURCE = local_source
    vehicle_config.DEFAULT_OUTPUT_ROOT = data / "vehicle_work"
    vehicle_config.RAW_ROOT = wild_root
    vehicle_config.DATA = data
    vehicle.SOURCE = local_source
    vehicle.RAW_ROOT = wild_root / "raw_predictions"
    vehicle.YOLO_ROOT = yolo_root
    vehicle_verify.YOLO_ROOT = yolo_root
    vehicle.confirm_vehicle_classes = lambda items, detections, min_iou=.5, score_threshold=.3, score_3d_threshold=.1: vehicle_confirm_classes(
        items, detections, min_iou=min_iou, score_threshold=score_threshold,
        score_3d_threshold=score_3d_threshold)
    vehicle.SAM_ROOT = data / "vehicle_work" / "instance_masks"
    vehicle.SAM_ASSOC_ROOT = data / "vehicle_work" / "association"
    vehicle.DEFAULT_OUTPUT_ROOT = data / "vehicle_work"
    (vehicle.SAM_ROOT).mkdir(parents=True, exist_ok=True)
    (vehicle.DEFAULT_OUTPUT_ROOT / "association").mkdir(parents=True, exist_ok=True)
    person_index, vehicle_index = [], []
    for ordinal,row in enumerate(rows,1):
        frame = int(Path(row["path"]).stem)
        stem = f"{frame:06d}"
        # Existing pedestrian processor's loader schema.
        pmeta = json.loads((sam_person / "metadata" / f"{stem}.json").read_text())
        pmeta["candidates"] = [dict(sam_instance_id=x["source_instance_id"], source_instance_id=x["source_instance_id"],
                                     prompt=x["prompt"], score=x["score"], bbox_xyxy=x["bbox_xyxy"],
                                     area_pixels=x["area_pixels"],track_id=x.get("track_id")) for x in pmeta["candidates"]]
        _json(sam_person / "metadata" / f"{stem}.json", pmeta)
        adapter_row = dict(row, source_frame_id=int(row.get("original_frame_id", row.get("frame_id", frame))))
        # Adapter symlink/copies only expose the paths expected by the bounded processors.
        pedestrian.load_wild_frame = lambda f, root, s=stem, p=wild_root: json.loads((p / "person_frames" / f"{s}.json").read_text())
        pedestrian.load_yolo_frame = lambda f, s=stem, p=data / "person_yolo": json.loads((p / f"{s}.json").read_text())
        person_result = pedestrian.process_frame(frame, adapter_row, sam_person, wild_root, person_root)
        person_npz = pedestrian.write_frame(person_result, person_root)
        person_index.append({"frame": stem, "original_frame_id": row.get("original_frame_id", row["frame_id"]),
                             "npz": str(person_npz.relative_to(person_root)),
                             "metadata": f"metadata/{stem}.json"})
        # Vehicle SAM dedup and exclusive-pixel caches expected by the existing processor.
        from associate_instance_masks import deduplicate, exclusive_pixel_masks
        masks, recs = _load_instances(sam_vehicle / "frames" / f"{stem}.npz",
                                      sam_vehicle / "metadata" / f"{stem}.json")
        recs = [dict(instance_id=int(r["source_instance_id"]), prompt=r["prompt"],
                     score=float(r["score"]), bbox_xyxy=r["bbox_xyxy"]) for r in recs]
        dedup_masks, dedup_records = deduplicate(list(masks), recs)
        exclusive, coverage = exclusive_pixel_masks(dedup_masks, image_shape=masks.shape[1:])
        np.savez_compressed(vehicle.DEFAULT_OUTPUT_ROOT / "association" / f"{stem}.npz",
                            masks=np.asarray(dedup_masks, bool).reshape((-1, *masks.shape[1:])),
                            exclusive_masks=np.asarray(exclusive, bool).reshape((-1, *masks.shape[1:])), coverage=coverage)
        _json(vehicle.SAM_ROOT / f"{stem}.json", {"image_source_sha256": row["original_payload_sha256"], "instances": dedup_records})
        _json(vehicle.SAM_ASSOC_ROOT / f"{stem}.json", {"source_image_sha256": row["original_payload_sha256"], "instances": dedup_records})
        # Adapter module uses a global path; no persistent project config is changed.
        vehicle_index.append(vehicle.process_frame(frame, vehicle_root, adapter_row))
        report_progress("instances",2*len(rows)+ordinal,3*len(rows),frame=row["frame_id"],detail="LiDAR instance labels",job=job)
    _json(person_root / "index.json", {"schema": "pedestrian_instance_labels_v3", "frames": person_index,
                                      "source_index_sha256": _sha256(job / "data/labels/zgc/index.json"),
                                      "manual_ground_truth": False, "added_to_training": False})
    _json(vehicle_root / "index.json", {"frames": vehicle_index, "manual_ground_truth": False,
                                        "added_to_training": False})


def run(job: Path, refresh=False):
    job = Path(job).resolve()
    index = job / "data/labels/zgc/index.json"
    payload = json.loads(index.read_text())
    rows = payload["rows"]
    if not rows:
        raise ValueError(f"No source rows in {index}")
    source, yolo_root, sam_person, sam_vehicle, wild_root = _prepare(job, rows)
    for local, row in enumerate(rows):
        if Path(row["path"]).stem != f"{local:06d}":
            raise ValueError("Fresh source paths must use dense local frame stems 000000..N-1")
        _source_aliases(job / "data/labels/zgc" / row["path"], source / row["path"])
    _run_models(job, rows, source, yolo_root, sam_person, sam_vehicle, wild_root, refresh=refresh)
    _materialize_processing_inputs(job, rows, source, sam_person, sam_vehicle, yolo_root, wild_root)
    _run_labelers(job, rows, source, yolo_root, sam_person, sam_vehicle, wild_root)
    shutil.copytree(sam_person,job / "data/person/raw_sam_person",dirs_exist_ok=True)
    return {"frames": len(rows), "person_root": str(job / "data/person"),
            "vehicle_root": str(job / "data/vehicle"), "status": "complete"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    result = run(parser.parse_args(argv).job)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
