"""One-frame independent YOLO vehicle-class check for WildDet3D candidates."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

from instance_label_config import (DATA, SOURCE, DEFAULT_OUTPUT_ROOT, RAW_ROOT,
                                   DISPLAY_SCORE_THRESHOLD, DISPLAY_SCORE_3D_THRESHOLD)
from label_geometry import passes_display_threshold, boxes_to_corners_lidar

YOLO_ROOT = DATA / "vehicle_yolo_mask_20261013/labels"
OUTPUT = DATA / "vehicle_failure_review_20261014/class_verification"
FRAME = "000402"
MIN_IOU = 0.5
VEHICLE_CLASS_NAMES = {"car", "bus", "truck"}


def box_iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.


def confirm_vehicle_classes(items, yolo_detections, min_iou=0.5,
                            score_threshold=DISPLAY_SCORE_THRESHOLD,
                            score_3d_threshold=DISPLAY_SCORE_3D_THRESHOLD):
    """One-to-one class confirmation for original score-gated WildDet candidates.

    Only WildDet candidates passing the configured score gates and YOLO
    detections whose existing class name is car/bus/truck enter Hungarian IoU
    matching. The function does not alter detector scores or infer object size.
    """
    wild = [item for item in items if str(item.get("category", "car")).lower() == "car"
            and float(item["score"]) >= score_threshold
            and float(item["score_3d"]) >= score_3d_threshold]
    yolo = [item for item in yolo_detections if str(item.get("name", "")).lower() in VEHICLE_CLASS_NAMES]
    metrics = np.zeros((len(wild), len(yolo)), dtype=np.float64)
    eligible = np.zeros_like(metrics, dtype=bool)
    for wi, candidate in enumerate(wild):
        for yi, detection in enumerate(yolo):
            value = box_iou(candidate["bbox_xyxy"], detection["bbox_xyxy"])
            metrics[wi, yi] = value
            eligible[wi, yi] = value >= min_iou
    matches = {}
    if len(wild) and len(yolo):
        rows, cols = linear_sum_assignment(np.where(eligible, 1.0 - metrics, 2.0))
        for wi, yi in zip(rows, cols):
            if eligible[wi, yi]:
                matches[int(wild[wi].get("index", wi))] = {
                    "yolo_detection_id": int(yolo[yi]["id"]),
                    "yolo_class_id": int(yolo[yi]["class_id"]),
                    "yolo_class_name": str(yolo[yi]["name"]),
                    "yolo_score": float(yolo[yi]["score"]),
                    "yolo_bbox_xyxy": list(map(float, yolo[yi]["bbox_xyxy"])),
                    "bbox_iou": float(metrics[wi, yi]),
                }
    records = []
    for wi, candidate in enumerate(wild):
        pred_id = int(candidate.get("index", wi))
        paired = matches.get(pred_id)
        records.append({"prediction_index": pred_id,
                        "status": "class_confirmed" if paired else "ignore_semantic_unconfirmed",
                        "wild_bbox_xyxy": list(map(float, candidate["bbox_xyxy"])),
                        "wild_score": float(candidate["score"]),
                        "wild_score_3d": float(candidate["score_3d"]),
                        "yolo_vehicle_match": paired,
                        "eligible_yolo_bbox_ious": [
                            {"yolo_detection_id": int(yolo[yi]["id"]),
                             "name": yolo[yi]["name"], "iou": float(metrics[wi, yi]),
                             "passes_iou_gate": bool(eligible[wi, yi])}
                            for yi in range(len(yolo))]})
    return records


def nearest_yolo_diagnostic(item, yolo_detections):
    vehicles = [d for d in yolo_detections if str(d.get("name", "")).lower() in VEHICLE_CLASS_NAMES]
    if not vehicles:
        return None
    detection = max(vehicles, key=lambda d: box_iou(item["bbox_xyxy"], d["bbox_xyxy"]))
    return {"yolo_detection_id": int(detection["id"]), "yolo_class_name": detection["name"],
            "bbox_iou_diagnostic_only": float(box_iou(item["bbox_xyxy"], detection["bbox_xyxy"])),
            "used_for_label_confirmation": False}


def _render_diagnostic(image, points_xyz, detections, yolo_detections, records, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    display = Image.fromarray(np.asarray(image.convert("RGB")).copy())
    draw = ImageDraw.Draw(display)
    record_by_id = {r["prediction_index"]: r for r in records}
    for detection in yolo_detections:
        x0, y0, x1, y1 = detection["bbox_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=(45, 185, 255), width=2)
        draw.text((x0 + 2, y0 + 2), f"YOLO #{detection['id']} {detection['name']} {detection['score']:.2f}",
                  fill=(45,185,255), stroke_width=1, stroke_fill=(0,0,0))
    for item in detections:
        index = int(item["index"])
        result = record_by_id.get(index)
        threshold_pass = passes_display_threshold(item["score"], item["score_3d"])
        if not threshold_pass:
            color, status = (240, 140, 36), "below WildDet gate"
        elif result and result["status"] == "class_confirmed":
            color, status = (50, 230, 105), "YOLO class matched"
        else:
            color, status = (255, 60, 60), "semantic unconfirmed"
        x0, y0, x1, y1 = item["bbox_xyxy"]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        draw.text((x0 + 2, max(0, y0 - 20)), f"Wild #{index} {item['score']:.2f}: {status}",
                  fill=color, stroke_width=1, stroke_fill=(0,0,0))
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=100)
    xyz = np.asarray(points_xyz, dtype=np.float64)
    valid = np.isfinite(xyz).all(1) & (np.abs(xyz[:, :2]) <= 20).all(1)
    ids = np.flatnonzero(valid)
    if len(ids) > 10000:
        ids = np.random.default_rng(20261014 + 402).choice(ids, 10000, replace=False)
    ax.scatter(xyz[ids, 1], xyz[ids, 0], s=.3, color="#aab5bc", alpha=.45, linewidths=0)
    ax.set_xlim(20,-20); ax.set_ylim(-20,20); ax.set_aspect("equal")
    ax.set_xlabel("LiDAR Y (m), left positive"); ax.set_ylabel("LiDAR X (m), forward positive")
    ax.set_title("000402 class-verification review: no 3D labels are added here")
    ax.grid(alpha=.2)
    fig.tight_layout();fig.canvas.draw()
    bev=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[...,:3]);plt.close(fig)
    bev=bev.resize((round(bev.width*display.height/bev.height),display.height),Image.Resampling.LANCZOS)
    sheet=Image.new("RGB",(display.width+bev.width,display.height+42),(24,24,24))
    sheet.paste(display,(0,42));sheet.paste(bev,(display.width,42))
    ImageDraw.Draw(sheet).text((12,12),"402: orange=beyond original Wild .6/.1 gate; red=class unconfirmed; green=YOLO car/bus/truck match",fill="white")
    sheet.save(output_path,quality=92,subsampling=0,optimize=True)


def main():
    out = OUTPUT / FRAME
    for folder in ("frames", "metadata", "support", "visualizations"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    source_rows = json.loads((SOURCE / "index.json").read_text())["rows"]
    row = next(row for row in source_rows if Path(row["path"]).stem == FRAME)
    source_path = SOURCE / "frames" / f"{FRAME}.npz"
    raw_path = RAW_ROOT / "raw_predictions" / f"{FRAME}.json"
    yolo_path = YOLO_ROOT / "metadata" / f"{FRAME}.json"
    yolo_policy_path = YOLO_ROOT / "policy.json"
    baseline_meta_path = DEFAULT_OUTPUT_ROOT / "refined" / f"{FRAME}.json"
    baseline_npz_path = DEFAULT_OUTPUT_ROOT / "refined" / f"{FRAME}.npz"
    raw = json.loads(raw_path.read_text())
    yolo_meta = json.loads(yolo_path.read_text())
    yolo_policy = json.loads(yolo_policy_path.read_text())
    baseline = json.loads(baseline_meta_path.read_text())
    detections = raw["items"]
    yolo_detections = yolo_meta["detections"]
    verification = confirm_vehicle_classes(detections, yolo_detections, min_iou=MIN_IOU)
    verification_by_id = {record["prediction_index"]: record for record in verification}
    baseline_by_id = {int(record["prediction_index"]): record for record in baseline["all_detections"]}
    all_records = []
    accepted_ids = []
    for item in detections:
        pred_id = int(item["index"])
        base = baseline_by_id.get(pred_id)
        class_result = verification_by_id.get(pred_id)
        if not passes_display_threshold(item["score"], item["score_3d"]):
            status = "ignore_below_wild_display_threshold"
        elif class_result is None or class_result["status"] != "class_confirmed":
            status = "ignore_semantic_unconfirmed"
        elif base is not None:
            status = base["status"]
            if status == "pseudo_label_experiment":
                accepted_ids.append(pred_id)
        else:
            status = "ignore_no_baseline_geometry"
        record = {"prediction_index": pred_id, "category": item.get("category", "car"),
                  "wild_score": float(item["score"]), "wild_score_3d": float(item["score_3d"]),
                  "bbox_xyxy": item["bbox_xyxy"], "passes_original_wild_display_gate": bool(passes_display_threshold(item["score"], item["score_3d"])),
                  "independent_vehicle_class_check": class_result,
                  "baseline_geometry_status_preserved": base.get("status") if base else None,
                  "status": status}
        if not passes_display_threshold(item["score"], item["score_3d"]):
            record["nearest_yolo_vehicle_diagnostic"] = nearest_yolo_diagnostic(item, yolo_detections)
        all_records.append(record)

    with np.load(baseline_npz_path) as archive:
        base_ids = archive["seed_ids"].astype(np.int64)
        base_boxes = archive["box_3d"].astype(np.float32)
        base_scores = archive["confidence"].astype(np.float32)
        base_score3d = archive["score_3d"].astype(np.float32)
        base_corners = archive["corners_3d_lidar"].astype(np.float32)
        K = archive["cam_intrinsic"].astype(np.float64)
        E = archive["cam_extrinsic"].astype(np.float64)
        timestamp = int(archive["timestamp_ns"])
    keep = np.array([int(seed_id) in accepted_ids for seed_id in base_ids], dtype=bool)
    boxes = base_boxes[keep]
    corners = base_corners[keep]
    seed_ids = base_ids[keep]
    confidence = base_scores[keep]
    score3d = base_score3d[keep]
    np.testing.assert_allclose(boxes_to_corners_lidar(boxes), corners, atol=2e-6)
    baseline_support_path = DEFAULT_OUTPUT_ROOT / "refined" / "support" / f"{FRAME}.npz"
    with np.load(baseline_support_path) as archive:
        ids = archive["point_indices"].astype(np.int64)
        offsets = archive["offsets"].astype(np.int64)
        base_support_seeds = archive["seed_ids"].astype(np.int64)
    selected_support_parts = []
    for i, seed_id in enumerate(base_support_seeds):
        if int(seed_id) in accepted_ids:
            selected_support_parts.append(ids[offsets[i]:offsets[i+1]])
    support_indices = np.concatenate(selected_support_parts) if selected_support_parts else np.empty(0,np.int64)
    support_offsets = np.zeros(len(seed_ids)+1,np.int64)
    if len(seed_ids):
        support_offsets[1:] = np.cumsum([len(x) for x in selected_support_parts])
    with np.load(source_path) as source:
        xyz = source["xyz"].astype(np.float64)
        uv = source["pixels_uv"].astype(np.float64)
    source_hash = row["original_payload_sha256"]
    final_npz = out / "frames" / f"{FRAME}.npz"
    np.savez_compressed(final_npz, box_3d=boxes, xy=boxes[:,:2], corners_3d_lidar=corners,
                        confidence=confidence, model_score=confidence, score_3d=score3d,
                        track_ids=np.full(len(seed_ids),-1,np.int64), seed_ids=seed_ids,
                        source_frame_id=np.array(402,np.int64), source_frame_id_string=np.array(row["frame_id"]),
                        frame_id=np.array(row["frame_id"]), timestamp_ns=np.array(timestamp),
                        is_pseudo_label=np.array(True), manual_gt=np.array(False), added_to_training=np.array(False),
                        display_score_threshold=np.array(DISPLAY_SCORE_THRESHOLD),
                        display_score_3d_threshold=np.array(DISPLAY_SCORE_3D_THRESHOLD),
                        ignore_bbox_xyxy=np.asarray([r["bbox_xyxy"] for r in all_records if r["status"] != "pseudo_label_experiment"],np.float32).reshape(-1,4),
                        ignore_prediction_indices=np.asarray([r["prediction_index"] for r in all_records if r["status"] != "pseudo_label_experiment"],np.int64),
                        support_point_indices=support_indices, support_offsets=support_offsets,
                        cam_intrinsic=K, cam_extrinsic=E)
    np.savez_compressed(out / "support" / f"{FRAME}.npz", point_indices=support_indices,
                        offsets=support_offsets, lidar_points=xyz[support_indices],
                        pixels_uv=uv[support_indices], seed_ids=seed_ids)
    counts = {}
    for record in all_records:
        counts[record["status"]] = counts.get(record["status"], 0) + 1
    report = {"frame": FRAME, "source_frame_id": row["frame_id"], "source_image_sha256": source_hash,
              "label_type": "single-frame independent vehicle-class verification experiment",
              "method": "Hungarian one-to-one 2D bbox IoU match from display-eligible WildDet3D car candidates to cached YOLO car/bus/truck detections; no 3D size rule or score threshold change",
              "wild_display_gate": {"combined_score": DISPLAY_SCORE_THRESHOLD, "score_3d": DISPLAY_SCORE_3D_THRESHOLD},
              "yolo_source": str(yolo_path), "yolo_policy": str(yolo_policy_path),
              "yolo_weight_sha256": yolo_policy["detector_sha256"],
              "yolo_class_names_allowed": sorted(VEHICLE_CLASS_NAMES), "minimum_bbox_iou": MIN_IOU,
              "yolo_detection_count": len(yolo_detections), "eligible_wild_detection_count": len(verification),
              "class_confirmed_wild_candidate_count": sum(x["status"] == "class_confirmed" for x in verification),
              "accepted_label_count": len(seed_ids), "status_counts": counts,
              "raw_low_score_vehicle_note": "Wild index 5 has combined score 0.3359 (<0.6); no label is emitted regardless of its nearby YOLO box.",
              "raw_prediction_source": str(raw_path), "baseline_geometry_source": str(baseline_npz_path),
              "detections": all_records, "output_npz": str(final_npz),
              "manual_gt": False, "added_to_training": False}
    (out / "metadata" / f"{FRAME}.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    sys.path.insert(0,str(DATA))
    from common import read_source_image
    _render_diagnostic(read_source_image(row),xyz,detections,yolo_detections,verification,
                       out / "visualizations" / f"{FRAME}.png")
    policy_matches = []
    for frame in ("000393","000402","000564"):
        raw_frame=json.loads((RAW_ROOT/"raw_predictions"/f"{frame}.json").read_text())
        yolo_frame=json.loads((YOLO_ROOT/"metadata"/f"{frame}.json").read_text())
        checked=confirm_vehicle_classes(raw_frame["items"],yolo_frame["detections"],min_iou=MIN_IOU)
        policy_matches.append({"frame":frame,"eligible_wild_candidates":len(checked),
                               "class_confirmed_matches":sum(x["status"]=="class_confirmed" for x in checked),
                               "matches":[{"wild_index":x["prediction_index"],"status":x["status"],
                                           "yolo":x["yolo_vehicle_match"]} for x in checked]})
    (OUTPUT / "three_frame_class_evidence.json").write_text(json.dumps({
        "policy": {"vehicle_names":sorted(VEHICLE_CLASS_NAMES),"wild_score":DISPLAY_SCORE_THRESHOLD,
                   "wild_score_3d":DISPLAY_SCORE_3D_THRESHOLD,"bbox_iou_min":MIN_IOU,
                   "assignment":"Hungarian one-to-one","no_frame_specific_rules":True},
        "frames":policy_matches},indent=2,ensure_ascii=False))
    (OUTPUT / "index.json").write_text(json.dumps({"kind":"independent_vehicle_class_verification_review",
        "frame_count":1,"manual_gt":False,"added_to_training":False,
        "rows":[{"frame":FRAME,"label_count":len(seed_ids),"metadata":f"{FRAME}/metadata/{FRAME}.json",
                 "npz":f"{FRAME}/frames/{FRAME}.npz","visualization":f"{FRAME}/visualizations/{FRAME}.png"}]},indent=2))
    print(json.dumps({"frame":FRAME,"label_count":len(seed_ids),"statuses":counts,
                      "three_frame_evidence":str(OUTPUT/"three_frame_class_evidence.json"),
                      "output":str(out)},indent=2))


if __name__ == "__main__":
    main()
