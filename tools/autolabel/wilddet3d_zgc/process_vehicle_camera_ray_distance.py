"""Single-frame fixed-pixel-centre, instance-distance-gated XY/yaw labels."""
import json
from pathlib import Path

import numpy as np

from instance_label_config import DATA, SOURCE, RAW_ROOT, DEFAULT_OUTPUT_ROOT
from associate_instance_masks import associate
from verify_vehicle_class_000402 import confirm_vehicle_classes, YOLO_ROOT
from refine_instance_support import valid_visible_ids, initial_upright_box, inside_training_roi
from center_distance_selection import fit_with_center_distance, InsufficientSupportError
from review_bev_pose import wrap_angle
from label_geometry import boxes_to_corners_lidar

SCORE_THRESHOLD = .3
SCORE_3D_THRESHOLD = .1
MIN_POINTS = 10
CLASS_IOU_THRESHOLD = .5
SAM_ROOT = DEFAULT_OUTPUT_ROOT / "instance_masks"
SAM_ASSOC_ROOT = DEFAULT_OUTPUT_ROOT / "association"
RAW_ROOT = RAW_ROOT / "raw_predictions"


def _center_uv(box, intrinsic, extrinsic):
    center_camera = np.asarray(box[:3], dtype=np.float64) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    homogeneous = intrinsic @ center_camera
    return homogeneous[:2] / homogeneous[2]


def process_frame(frame, output_root, source_row):
    """Use all owned instance returns; one helper gate chooses final support."""
    name = f"{int(frame):06d}"
    output_root = Path(output_root)
    for folder in ("refined/support", "association", "reports"):
        (output_root / folder).mkdir(parents=True, exist_ok=True)
    source_hash = source_row["original_payload_sha256"]
    raw_path = RAW_ROOT / f"{name}.json"
    yolo_path = YOLO_ROOT / "metadata" / f"{name}.json"
    sam_meta_path = SAM_ROOT / f"{name}.json"
    sam_assoc_path = SAM_ASSOC_ROOT / f"{name}.json"
    raw_header = json.loads(raw_path.read_text())
    raw = raw_header["items"]
    if raw_header.get("source_payload_sha256") != source_hash:
        raise ValueError(f"WildDet3D source SHA mismatch for {name}")
    yolo_meta = json.loads(yolo_path.read_text())
    if yolo_meta.get("frame_id") != source_row.get("frame_id"):
        raise ValueError(f"YOLO frame ID mismatch for {name}")
    yolo = yolo_meta["detections"]
    class_records = confirm_vehicle_classes(raw, yolo, min_iou=CLASS_IOU_THRESHOLD,
                                             score_threshold=SCORE_THRESHOLD)
    class_records_by_id = {int(record["prediction_index"]): record for record in class_records}
    class_confirmed_ids = {index for index, record in class_records_by_id.items()
                           if record["status"] == "class_confirmed"}
    candidate_items = [item for item in raw if int(item["index"]) in class_confirmed_ids]
    sam_meta = json.loads(sam_meta_path.read_text())
    if sam_meta.get("image_source_sha256") != source_hash:
        raise ValueError(f"SAM instance source SHA mismatch for {name}")
    old_assoc = json.loads(sam_assoc_path.read_text())
    if old_assoc.get("source_image_sha256") != source_hash:
        raise ValueError(f"SAM association source SHA mismatch for {name}")
    with np.load(DEFAULT_OUTPUT_ROOT / "association" / f"{name}.npz") as archive:
        masks = archive["masks"].astype(bool)
        exclusive = archive["exclusive_masks"].astype(bool)
    associations, _, _ = associate(candidate_items, masks, old_assoc["instances"])
    association_by_id = {int(record["prediction_index"]): record for record in associations}
    association_report = {
        "frame": name, "source_frame_id": source_row["frame_id"], "source_image_sha256": source_hash,
        "score_threshold": SCORE_THRESHOLD, "score_3d_threshold": SCORE_3D_THRESHOLD,
        "class_iou_threshold": CLASS_IOU_THRESHOLD,
        "instances": old_assoc["instances"], "associations": associations,
        "class_verification": class_records,
        "support_policy": "all exclusive instance-mask LiDAR points inside the WildDet bbox; one centre-distance gate decides final solver support",
        "no_dbscan_or_cluster_selection": True,
    }
    assoc_out = output_root / "association" / f"{name}.json"
    assoc_out.write_text(json.dumps(association_report, indent=2, ensure_ascii=False))

    with np.load(SOURCE / "frames" / f"{name}.npz") as archive:
        source = {key: archive[key] for key in archive.files}
    xyz, uv, _, visible_ids, pixels = valid_visible_ids(source, *masks.shape[1:])
    K = source["cam_intrinsic"].astype(np.float64)
    E = source["cam_extrinsic"].astype(np.float64)
    inverse = np.linalg.inv(E)
    owner = np.full(masks.shape[1:], -1, dtype=np.int32)
    for instance_id, instance_mask in enumerate(exclusive):
        owner[instance_mask] = instance_id
    point_owner = owner[pixels[:, 1], pixels[:, 0]]
    if len(visible_ids[point_owner >= 0]) != len(np.unique(visible_ids[point_owner >= 0])):
        raise AssertionError(f"LiDAR return assigned more than once in {name}")

    records, accepted, selected_parts = [], [], []
    for item in raw:
        seed = int(item["index"])
        class_record = class_records_by_id.get(seed)
        yolo_match = class_record.get("yolo_vehicle_match") if class_record else None
        record = {"prediction_index": seed, "wild_score": float(item["score"]),
                  "score_3d": float(item["score_3d"]), "category": item.get("category", "car"),
                  "class_name": yolo_match["yolo_class_name"] if yolo_match else None,
                  "class_verification": yolo_match, "bbox_xyxy": item["bbox_xyxy"]}
        records.append(record)
        if not (float(item["score"]) >= SCORE_THRESHOLD
                and float(item["score_3d"]) >= SCORE_3D_THRESHOLD):
            record["status"] = "ignore_below_score_gate"
            continue
        if seed not in class_confirmed_ids:
            record["status"] = "ignore_semantic_unconfirmed"
            continue
        association = association_by_id.get(seed)
        if association is None or association.get("status") != "matched":
            record["status"] = "ignore_instance_association"
            if association is not None:
                record["association"] = association
            continue
        instance_id = int(association["instance_id"])
        bbox = np.asarray(item["bbox_xyxy"], dtype=np.float64)
        point_ids = visible_ids[point_owner == instance_id]
        inside_detector_bbox = ((uv[point_ids] >= bbox[:2]).all(1)
                                & (uv[point_ids] < bbox[2:]).all(1))
        all_instance_points = point_ids[inside_detector_bbox]
        record.update({"sam_instance_id": instance_id,
                       "sam_bbox_xyxy": old_assoc["instances"][instance_id]["bbox_xyxy"],
                       "association": association,
                       "input_exclusive_instance_point_count": int(len(all_instance_points))})
        if len(all_instance_points) < MIN_POINTS:
            record["status"] = "ignore_insufficient_instance_support"
            continue

        prior, _, _ = initial_upright_box(item, inverse)
        native_vectors = np.asarray(item["corners_cam"], dtype=np.float64) @ inverse[:3, :3].T
        raw_long_axis = native_vectors[0] - native_vectors[2]
        prior[6] = np.arctan2(raw_long_axis[1], raw_long_axis[0])
        mask_bbox = np.asarray(old_assoc["instances"][instance_id]["bbox_xyxy"], dtype=np.float64)
        try:
            box, diagnostics, keep = fit_with_center_distance(
                prior, xyz[all_instance_points], K, E, mask_bbox, masks.shape[1:][::-1])
        except InsufficientSupportError as error:
            record.update(status="ignore_insufficient_center_distance_support",
                          center_distance_selection=error.diagnostics)
            continue
        except (ValueError, RuntimeError, FloatingPointError) as error:
            record.update(status="ignore_camera_ray_fit_failure",
                          error=f"{type(error).__name__}: {error}")
            continue
        keep = np.asarray(keep, dtype=bool).reshape(-1)
        if keep.shape != (len(all_instance_points),):
            raise AssertionError(f"Distance helper keep mask does not align to instance points in {name}/{seed}")
        selected_points = xyz[all_instance_points[keep]]
        if len(selected_points) < MIN_POINTS:
            record.update(status="ignore_insufficient_center_distance_support",
                          center_distance_selection=diagnostics)
            continue
        pixel_before = _center_uv(prior, K, E)
        pixel_after = _center_uv(box, K, E)
        pixel_shift = float(np.linalg.norm(pixel_after - pixel_before))
        yaw_delta = float(wrap_angle(box[6] - prior[6]))
        if pixel_shift > 1e-3:
            raise AssertionError(f"Fixed input pixel centre shifted {pixel_shift:.6g}px in {name}/{seed}")
        if abs(yaw_delta) > np.pi / 2 + 1e-6:
            raise AssertionError(f"Directed yaw moved beyond 90 degrees in {name}/{seed}")
        if not np.allclose(box[3:6], prior[3:6], atol=2e-6, rtol=0):
            raise AssertionError(f"Frozen LWH changed in {name}/{seed}")
        record.update(initial_box_3d=prior.tolist(), box_3d=box.tolist(),
                      fixed_pixel_center_uv=pixel_before.tolist(), final_pixel_center_uv=pixel_after.tolist(),
                      center_pixel_shift_px=pixel_shift,
                      initial_directed_yaw_rad=float(prior[6]), yaw_adjustment_rad=yaw_delta,
                      diagnostics=diagnostics,
                      support_count_after_distance_gate=int(len(selected_points)))
        if not inside_training_roi(box):
            record["status"] = "ignore_outside_roi"
            continue
        record["status"] = "review_label"
        accepted.append((item, box, diagnostics, prior[6], yaw_delta, all_instance_points[keep]))
        selected_parts.append(all_instance_points[keep])

    support_indices = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    if len(support_indices) != len(np.unique(support_indices)):
        raise AssertionError(f"Cross-label exclusive support reuse in {name}")
    accepted.sort(key=lambda entry: int(entry[0]["index"]))
    boxes = np.asarray([entry[1] for entry in accepted], dtype=np.float32).reshape(-1, 7)
    seeds = np.asarray([int(entry[0]["index"]) for entry in accepted], dtype=np.int64)
    corners = boxes_to_corners_lidar(boxes).astype(np.float32)
    scores = np.asarray([entry[0]["score"] for entry in accepted], dtype=np.float32)
    scores3d = np.asarray([entry[0]["score_3d"] for entry in accepted], dtype=np.float32)
    initial_yaw = np.asarray([entry[3] for entry in accepted], dtype=np.float32)
    yaw_adjustment = np.asarray([entry[4] for entry in accepted], dtype=np.float32)
    support_offsets = np.zeros(len(seeds) + 1, dtype=np.int64)
    if accepted:
        support_offsets[1:] = np.cumsum([len(entry[5]) for entry in accepted])
    np.testing.assert_allclose(boxes_to_corners_lidar(boxes), corners, atol=2e-6, rtol=0)

    output_npz = output_root / "refined" / f"{name}.npz"
    np.savez_compressed(output_npz, box_3d=boxes, xy=boxes[:, :2], corners_3d_lidar=corners,
                        seed_ids=seeds, confidence=scores, model_score=scores, score_3d=scores3d,
                        initial_directed_yaw=initial_yaw, yaw_adjustment=yaw_adjustment,
                        support_point_indices=support_indices, support_offsets=support_offsets,
                        cam_intrinsic=K, cam_extrinsic=E,
                        display_score_threshold=np.array(SCORE_THRESHOLD),
                        display_score_3d_threshold=np.array(SCORE_3D_THRESHOLD),
                        manual_gt=np.array(False), added_to_training=np.array(False),
                        review_only=np.array(False), config_approved=np.array(True),
                        is_pseudo_label=np.array(True), source_frame_id=np.array(int(source_row["source_frame_id"])),
                        source_frame_id_string=np.array(source_row["frame_id"]), frame_id=np.array(source_row["frame_id"]),
                        timestamp_ns=np.array(int(source["lidar_timestamp_ns"])),
                        class_names=np.asarray([class_records_by_id[int(entry[0]["index"])]["yolo_vehicle_match"]["yolo_class_name"]
                                                for entry in accepted],dtype=np.str_))
    support_path = output_root / "refined" / "support" / f"{name}.npz"
    np.savez_compressed(support_path, point_indices=support_indices, offsets=support_offsets,
                        seed_ids=seeds, lidar_points=xyz[support_indices], pixels_uv=uv[support_indices])
    source_class_records = {int(record["prediction_index"]): record for record in class_records}
    class_names = [source_class_records[int(seed)]["yolo_vehicle_match"]["yolo_class_name"] for seed in seeds]
    statuses = {status: sum(record["status"] == status for record in records)
                for status in sorted({record["status"] for record in records})}
    metadata = {"frame": name, "source_frame_id": source_row["frame_id"],
                "source_frame_numeric_id": int(source_row["source_frame_id"]),
                "source_image_sha256": source_hash, "timestamp_ns": int(source["lidar_timestamp_ns"]),
                "label_type": "camera-ray-fixed-pixel XY/yaw vehicle pseudo-label; center-distance support gate",
                "score_threshold": SCORE_THRESHOLD, "score_3d_threshold": SCORE_3D_THRESHOLD,
                "class_match_iou_threshold": CLASS_IOU_THRESHOLD,
                "class_names": class_names, "accepted_count": int(len(seeds)),
                "all_detections": records, "status_counts": statuses,
                "support_selection": {"mode":"all valid exclusive instance mask returns inside WildDet bbox, then one center-distance gate",
                    "dbscan_used":False,"repeat_selection":False,
                    "minimum_retained_points":MIN_POINTS,"helper":"center_distance_selection.fit_with_center_distance"},
                "fixed_pixel_center":True,"fixed_lwh":True,"ground_z_override":False,
                "manual_gt":False,"added_to_training":False,"config_approved":True,
                "requires_user_review_before_batch":False,
                "raw_prediction_source":str(raw_path),"yolo_class_source":str(yolo_path),
                "sam_instance_source":str(sam_meta_path),"association_source":str(assoc_out)}
    metadata_path=output_root/"refined"/f"{name}.json"
    metadata_path.write_text(json.dumps(metadata,indent=2,ensure_ascii=False))
    return {"frame":name,"accepted_count":int(len(seeds)),"seed_ids":seeds.tolist(),"statuses":statuses,
            "source_image_sha256":source_hash,"output_npz":str(output_npz),"metadata":str(metadata_path)}
