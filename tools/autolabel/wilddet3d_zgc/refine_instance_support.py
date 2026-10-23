"""Generate labels from uniquely associated SAM-instance LiDAR supports."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.cluster import DBSCAN

from instance_label_config import (DATA, SOURCE, DEFAULT_OUTPUT_ROOT, RAW_ROOT,
                                   MIN_SELECTED_POINTS, DBSCAN_EPS_METERS,
                                   DBSCAN_MIN_SAMPLES, TRAINING_ROI_XY,
                                   TRAINING_ROI_Z, DISPLAY_SCORE_THRESHOLD,
                                   DISPLAY_SCORE_3D_THRESHOLD, HUBER_DELTA_METERS,
                                   OBJECTIVE_SURFACE_WEIGHT, OBJECTIVE_FIRST_RAY_WEIGHT)
from label_geometry import boxes_to_corners_lidar, passes_display_threshold
from refine_mask_depth_000372 import box_metrics, huber

ROOT = DEFAULT_OUTPUT_ROOT
SAM = ROOT / "instance_masks"
RAW = RAW_ROOT / "raw_predictions"
ASSOCIATION_OUT = ROOT / "association"
OUT = ROOT / "refined"
NATIVE_TO_CUBE = np.array([2, 0, 4, 6, 3, 1, 5, 7])
MIN_POINTS = MIN_SELECTED_POINTS
ROI_XY = TRAINING_ROI_XY
ROI_Z = TRAINING_ROI_Z


def inside_training_roi(box):
    return (ROI_XY[0] <= box[0] < ROI_XY[1]
            and ROI_XY[0] <= box[1] < ROI_XY[1]
            and ROI_Z[0] <= box[2] < ROI_Z[1])


def load_frame(frame):
    with np.load(SOURCE / "frames" / f"{frame}.npz") as archive:
        return {key: archive[key] for key in archive.files}


def valid_visible_ids(source, height, width):
    xyz = source["xyz"].astype(np.float64)
    uv = source["pixels_uv"].astype(np.float64)
    depth = source["camera_depth"].astype(np.float64)
    valid = (source["visible"].astype(bool) & np.isfinite(xyz).all(1)
             & np.isfinite(uv).all(1) & np.isfinite(depth) & (depth > .1))
    ids = np.flatnonzero(valid)
    pix = np.floor(uv[ids]).astype(np.int64)
    inside = (pix[:, 0] >= 0) & (pix[:, 0] < width) & (pix[:, 1] >= 0) & (pix[:, 1] < height)
    return xyz, uv, depth, ids[inside], pix[inside]


def initial_upright_box(item, inverse):
    camera = np.asarray(item["box_camera_center_wlh_quaternion"], dtype=np.float64)
    native = np.asarray(item["corners_cam"], dtype=np.float64)
    corners_lidar_full = native[NATIVE_TO_CUBE] @ inverse[:3, :3].T + inverse[:3, 3]
    center_lidar = camera[:3] @ inverse[:3, :3].T + inverse[:3, 3]
    w, length, height = map(float, camera[3:6])
    axes = np.column_stack(((corners_lidar_full[1] - corners_lidar_full[0]) / length,
                            (corners_lidar_full[4] - corners_lidar_full[0]) / w,
                            (corners_lidar_full[3] - corners_lidar_full[0]) / height))
    yaw = float(np.arctan2(axes[1, 0], axes[0, 0]) % np.pi)
    box = np.array([*center_lidar, length, w, height, yaw], dtype=np.float64)
    upright_corners = boxes_to_corners_lidar(box[None])[0]
    upright_axes = np.column_stack(((upright_corners[1] - upright_corners[0]) / length,
                                    (upright_corners[4] - upright_corners[0]) / w,
                                    (upright_corners[3] - upright_corners[0]) / height))
    np.testing.assert_allclose(upright_axes.T @ upright_axes, np.eye(3), atol=1e-10)
    return box, upright_corners, upright_axes


def fit_depth(box, upright_corners, axes, points, inverse, K):
    center_lidar = box[:3]
    center_camera = center_lidar @ np.linalg.inv(inverse)[:3, :3].T + np.linalg.inv(inverse)[:3, 3]
    initial_z = float(center_camera[2])
    ray = center_camera / initial_z
    offsets = upright_corners - center_lidar
    half = box[3:6] / 2
    args = (ray, inverse, offsets, axes, half, points)

    def objective(z):
        _, surface, ray_error, _, _ = box_metrics(float(z), *args)
        return float(OBJECTIVE_SURFACE_WEIGHT * np.mean(huber(surface, HUBER_DELTA_METERS))
                     + OBJECTIVE_FIRST_RAY_WEIGHT * np.mean(huber(ray_error, HUBER_DELTA_METERS)))

    bounds = (.2, max(initial_z * 2, initial_z + np.linalg.norm(half) + 2.))
    grid = np.linspace(*bounds, 241)
    costs = np.array([objective(z) for z in grid])
    index = int(costs.argmin())
    left, right = grid[max(0, index - 1)], grid[min(len(grid) - 1, index + 1)]
    if right <= left:
        left, right = bounds
    fit = minimize_scalar(objective, bounds=(left, right), method="bounded", options={"xatol": 1e-7})

    def geometry(z):
        center = (float(z) * ray) @ inverse[:3, :3].T + inverse[:3, 3]
        center_camera = float(z) * ray
        center_uvw = K @ center_camera
        center_uv = center_uvw[:2] / center_uvw[2]
        final_box = np.array([*center, *box[3:6], box[6]], dtype=np.float64)
        corners = boxes_to_corners_lidar(final_box[None])[0]
        _, surface, ray_error, outside, hit = box_metrics(float(z), *args)
        return final_box, corners, {
            "camera_z_m": float(z), "center_lidar_xyz": center.tolist(),
            "center_uv": center_uv.tolist(),
            "objective": objective(z), "median_abs_surface_distance_m": float(np.median(np.abs(surface))),
            "median_abs_ray_error_m": float(np.median(np.abs(ray_error))),
            "points_inside_final_upright_box": int((outside < 1e-5).sum()),
            "first_ray_hits": int(hit.sum()),
        }

    before_box, before_corners, before = geometry(initial_z)
    after_box, after_corners, after = geometry(float(fit.x))
    np.testing.assert_allclose(before["center_uv"], after["center_uv"], atol=1e-9)
    # The final label geometry is always decoded from its upright 7D box.
    np.testing.assert_allclose(boxes_to_corners_lidar(before_box[None])[0], before_corners, atol=1e-10)
    np.testing.assert_allclose(boxes_to_corners_lidar(after_box[None])[0], after_corners, atol=1e-10)
    origin = inverse[:3, 3]
    if np.linalg.norm(np.cross(before_box[:3] - origin, after_box[:3] - origin)) >= 1e-6:
        raise AssertionError("Refined upright-box center moved off its fixed camera ray")
    return before_box, before_corners, before, after_box, after_corners, after, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="*", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    rows = json.loads((SOURCE / "index.json").read_text())["rows"]
    by_frame = {int(Path(row["path"]).stem): row for row in rows}
    frame_ids = sorted(by_frame) if args.frames is None else sorted(set(args.frames))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "support").mkdir(exist_ok=True)
    rows_out = []
    skipped = []
    for frame in frame_ids:
        name = f"{frame:06d}"
        meta_path, npz_path = OUT / f"{name}.json", OUT / f"{name}.npz"
        if not args.no_resume and meta_path.is_file() and npz_path.is_file():
            existing = json.loads(meta_path.read_text())
            if existing.get("source_image_sha256") == by_frame[frame].get("original_payload_sha256"):
                rows_out.append({"frame": name, "accepted": existing["accepted_count"], "resumed": True})
                skipped.append(name)
                continue
        source = load_frame(name)
        K = source["cam_intrinsic"].astype(np.float64)
        E = source["cam_extrinsic"].astype(np.float64)
        inverse = np.linalg.inv(E)
        assoc = json.loads((ASSOCIATION_OUT / f"{name}.json").read_text())
        raw = json.loads((RAW / f"{name}.json").read_text())
        source_hash = by_frame[frame].get("original_payload_sha256")
        for label, report in (("instance mask", json.loads((SAM / f"{name}.json").read_text())),
                              ("association", assoc), ("raw WildDet3D", raw)):
            hash_key = "image_source_sha256" if label == "instance mask" else ("source_payload_sha256" if label == "raw WildDet3D" else "source_image_sha256")
            if report.get(hash_key) != source_hash:
                raise ValueError(f"{label} source image SHA-256 mismatch for {name}")
        detection_by_id = {int(item["index"]): item for item in raw["items"]}
        with np.load(ASSOCIATION_OUT / f"{name}.npz") as masks_file:
            exclusive_masks = masks_file["exclusive_masks"].astype(bool)
        height, width = exclusive_masks.shape[1:]
        xyz, uv, lidar_depth, visible_ids, pixels = valid_visible_ids(source, height, width)
        pixel_owner = np.full((height, width), -1, dtype=np.int32)
        for inst, mask in enumerate(exclusive_masks):
            pixel_owner[mask] = inst
        point_owner = pixel_owner[pixels[:, 1], pixels[:, 0]]
        point_ids = visible_ids[point_owner >= 0]
        owner_ids = point_owner[point_owner >= 0]
        if len(point_ids) != len(np.unique(point_ids)):
            raise AssertionError("Projected LiDAR point assigned to multiple distinct SAM instances")
        records, accepted, all_support = [], [], []
        for association in assoc["associations"]:
            index = int(association["prediction_index"])
            item = detection_by_id[index]
            record = {"prediction_index": index, "model_score": float(item["score"]),
                      "score_3d": float(item["score_3d"]), "bbox_xyxy": item["bbox_xyxy"],
                      "passes_display_threshold": passes_display_threshold(item["score"], item["score_3d"]),
                      "association": association, "status": "ignore_instance_association"}
            if association["status"] != "matched":
                records.append(record)
                continue
            instance_id = int(association["instance_id"])
            # Instance mask AND detection bbox; ambiguous overlap pixels are absent from exclusive_masks.
            own_rows = np.flatnonzero(owner_ids == instance_id)
            candidate_ids = point_ids[own_rows]
            bbox = np.asarray(item["bbox_xyxy"], dtype=np.float64)
            inside_det = ((uv[candidate_ids] >= bbox[:2]).all(1) & (uv[candidate_ids] < bbox[2:]).all(1))
            candidate_ids = candidate_ids[inside_det]
            initial_count = len(candidate_ids)
            cluster_report = []
            if len(candidate_ids) >= 3:
                cluster_ids = DBSCAN(eps=DBSCAN_EPS_METERS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xyz[candidate_ids])
                cluster_report = [{"cluster": int(g), "points": int(np.count_nonzero(cluster_ids == g))}
                                  for g in np.unique(cluster_ids)]
                clusters = [(int(np.count_nonzero(cluster_ids == g)), int(g)) for g in np.unique(cluster_ids) if g >= 0]
                if clusters:
                    _, selected_cluster = max(clusters)
                    candidate_ids = candidate_ids[cluster_ids == selected_cluster]
                else:
                    candidate_ids = np.empty(0, dtype=np.int64)
            record.update({"sam_instance_id": instance_id, "instance_owned_visible_points_inside_detection": initial_count,
                           "cluster_counts_including_noise": cluster_report, "selected_point_count": int(len(candidate_ids))})
            if len(candidate_ids) < MIN_POINTS:
                record["status"] = "ignore_insufficient_exclusive_lidar_support"
                records.append(record)
                continue
            box, upright_corners, axes = initial_upright_box(item, inverse)
            before_box, before_corners, before, after_box, after_corners, after, fit = fit_depth(
                box, upright_corners, axes, xyz[candidate_ids], inverse, K)
            record.update({"initial_box_3d_lidar": before_box.tolist(), "before": before,
                           "after_box_3d_lidar": after_box.tolist(), "after": after,
                           "optimizer_success": bool(fit.success),
                           "objective": f"surface weight={OBJECTIVE_SURFACE_WEIGHT:g}, first-ray weight={OBJECTIVE_FIRST_RAY_WEIGHT:g}; Huber delta={HUBER_DELTA_METERS:g}m",
                           "ray_geometry_limit": "A solid cuboid first-hit ray is only a control approximation; car body surfaces are curved/open and a fixed monocular center ray may be wrong."})
            if not fit.success or after["objective"] > before["objective"] + 1e-8:
                record["status"] = "ignore_optimizer_failure"
            elif not inside_training_roi(after_box):
                record["status"] = "ignore_outside_training_roi"
            else:
                record["status"] = "pseudo_label_experiment"
                accepted.append((index, item, record, after_box, after_corners, candidate_ids))
                all_support.append(candidate_ids)
            records.append(record)

        # Accepted supports must remain disjoint across instances and predictions.
        if all_support:
            flat = np.concatenate(all_support)
            if len(flat) != len(np.unique(flat)):
                raise AssertionError("A LiDAR support point appears in more than one accepted instance label")
        accepted.sort(key=lambda v: v[0])
        boxes = np.asarray([x[3] for x in accepted], dtype=np.float32).reshape(-1, 7)
        corners = boxes_to_corners_lidar(boxes).astype(np.float32)
        if len(boxes):
            expected = np.asarray([x[4] for x in accepted], dtype=np.float32)
            np.testing.assert_allclose(corners, expected, atol=2e-6)
        out_npz = npz_path
        support_offsets = np.zeros(len(accepted) + 1, dtype=np.int64)
        if accepted:
            support_offsets[1:] = np.cumsum([len(x[5]) for x in accepted])
            support_indices = np.concatenate([x[5] for x in accepted]).astype(np.int64)
        else:
            support_indices = np.empty(0, dtype=np.int64)
        scores = np.asarray([x[1]["score"] for x in accepted], dtype=np.float32)
        score3d = np.asarray([x[1]["score_3d"] for x in accepted], dtype=np.float32)
        seed_ids = np.asarray([x[0] for x in accepted], dtype=np.int64)
        source_id = str(source_hash and by_frame[frame].get("frame_id", name))
        tmp_npz_path = OUT / f"{name}.tmp.npz"
        np.savez_compressed(tmp_npz_path, box_3d=boxes, xy=boxes[:, :2], corners_3d_lidar=corners,
                            confidence=scores, model_score=scores, score_3d=score3d,
                            track_ids=np.full(len(accepted), -1, dtype=np.int64), seed_ids=seed_ids,
                            source_frame_id=np.array(frame, dtype=np.int64),
                            source_frame_id_string=np.array(source_id), frame_id=np.array(source_id),
                            timestamp_ns=np.array(int(source["lidar_timestamp_ns"])), is_pseudo_label=np.array(True),
                            manual_gt=np.array(False), added_to_training=np.array(False),
                            display_score_threshold=np.array(DISPLAY_SCORE_THRESHOLD),
                            display_score_3d_threshold=np.array(DISPLAY_SCORE_3D_THRESHOLD),
                            ignore_bbox_xyxy=np.asarray([r["bbox_xyxy"] for r in records if r.get("status") != "pseudo_label_experiment"], dtype=np.float32).reshape(-1,4),
                            ignore_prediction_indices=np.asarray([r["prediction_index"] for r in records if r.get("status") != "pseudo_label_experiment"], dtype=np.int64),
                            support_point_indices=support_indices, support_offsets=support_offsets,
                            cam_intrinsic=K, cam_extrinsic=E)
        tmp_npz_path.replace(out_npz)
        support_npz_path = OUT / "support" / f"{name}.npz"
        np.savez_compressed(support_npz_path, point_indices=support_indices, offsets=support_offsets,
                            lidar_points=xyz[support_indices], pixels_uv=uv[support_indices], seed_ids=seed_ids)
        report = {"frame": name, "status": "complete", "label_type": "confirmed instance-associated pseudo-ground-truth experiment",
                  "source_frame_id": by_frame[frame].get("frame_id"), "source_image_sha256": source_hash,
                  "raw_prediction_source": str(RAW / f"{name}.json"),
                  "source_instance_masks": str(SAM / f"{name}.npz"),
                  "association_report": str(ASSOCIATION_OUT / f"{name}.json"),
                  "LiDAR_point_owner_unique": True, "final_box_geometry": "corners_3d_lidar decoded from saved box_3d only",
                  "box_3d_fields": ["x", "y", "z", "length", "width", "height", "yaw_long_axis_mod_pi"],
                  "confidence_meaning": "Raw WildDet3D combined score; not a calibrated probability and may exceed 1",
                  "display_thresholds": {"combined_score": DISPLAY_SCORE_THRESHOLD,
                                         "score_3d": DISPLAY_SCORE_3D_THRESHOLD},
                  "point_selection": f"Exclusive SAM instance pixels inside associated detector bbox; largest DBSCAN cluster eps={DBSCAN_EPS_METERS}m/min_samples={DBSCAN_MIN_SAMPLES}; min {MIN_POINTS} selected points",
                  "objective": f"Equal weights: final-upright-cuboid surface Huber + first LiDAR-ray entry Huber (delta={HUBER_DELTA_METERS}m); optimize camera Z along fixed camera ray",
                  "unmatched_and_unlabeled_regions": "ignore, not verified negatives",
                  "training_roi_gate": {"x_m": list(ROI_XY), "y_m": list(ROI_XY), "z_m": list(ROI_Z), "upper_bounds_exclusive": True},
                  "all_detections": records, "accepted_count": len(accepted), "output_npz": str(out_npz),
                  "support_npz": str(support_npz_path), "manual_gt": False, "added_to_training": False}
        tmp_json_path = OUT / f"{name}.tmp.json"
        tmp_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp_json_path.replace(meta_path)
        rows_out.append({"frame": name, "accepted": len(accepted),
                         "statuses": {key: sum(r["status"] == key for r in records) for key in sorted({r['status'] for r in records})}})
        print(json.dumps(rows_out[-1], ensure_ascii=False), flush=True)
    (OUT / "summary.json").write_text(json.dumps({"frames": rows_out, "frame_count": len(rows_out),
                                                   "skipped_valid": len(skipped), "status": "complete"}, indent=2))


if __name__ == "__main__":
    main()
