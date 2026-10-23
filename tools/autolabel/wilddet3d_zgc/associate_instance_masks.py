"""Conservatively associate saved SAM3 vehicle instances to WildDet boxes.

Only the five diagnostic frames are processed. Distinct SAM masks and all raw
detections remain recorded; unassigned masks are diagnostics, never negatives.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from instance_label_config import (DATA, SOURCE, DEFAULT_OUTPUT_ROOT, RAW_ROOT,
                                   DISPLAY_SCORE_THRESHOLD, DISPLAY_SCORE_3D_THRESHOLD,
                                   PROMPT_DEDUP_MASK_IOU, PROMPT_DEDUP_BOX_IOU,
                                   ASSOC_BOX_IOU_MIN, ASSOC_MASK_COVERAGE_MIN,
                                   ASSOC_UTILITY_MARGIN_MIN)

ROOT = DEFAULT_OUTPUT_ROOT
SAM = ROOT / "instance_masks"
RAW = RAW_ROOT / "raw_predictions"
OUT = ROOT / "association"
FRAMES = []
DEDUP_MASK_IOU = PROMPT_DEDUP_MASK_IOU
DEDUP_BOX_IOU = PROMPT_DEDUP_BOX_IOU
MIN_ASSOCIATION_BOX_IOU = ASSOC_BOX_IOU_MIN
MIN_ASSOCIATION_MASK_COVERAGE = ASSOC_MASK_COVERAGE_MIN
MIN_UTILITY_MARGIN = ASSOC_UTILITY_MARGIN_MIN


def box_intersection(a, b):
    return max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(0., min(a[3], b[3]) - max(a[1], b[1]))


def box_iou(a, b):
    intersection = box_intersection(a, b)
    area_a = max(0., a[2] - a[0]) * max(0., a[3] - a[1])
    area_b = max(0., b[2] - b[0]) * max(0., b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.


def mask_iou(a, b):
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.


def bbox(mask):
    yy, xx = np.nonzero(mask)
    return [float(xx.min()), float(yy.min()), float(xx.max() + 1), float(yy.max() + 1)] if len(xx) else [0., 0., 0., 0.]


def deduplicate(masks, records):
    """Union only near-identical cross-prompt masks; preserve prompt provenance."""
    parent = list(range(len(masks)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            if records[i]["prompt"] == records[j]["prompt"]:
                continue
            if (mask_iou(masks[i], masks[j]) >= DEDUP_MASK_IOU
                    and box_iou(records[i]["bbox_xyxy"], records[j]["bbox_xyxy"]) >= DEDUP_BOX_IOU):
                union(i, j)
    groups = {}
    for i in range(len(masks)):
        groups.setdefault(find(i), []).append(i)
    result_masks, result_records = [], []
    for group in groups.values():
        # Select the highest-scoring prompt variant; never OR distinct masks.
        representative = max(group, key=lambda i: records[i]["score"])
        result_masks.append(masks[representative].astype(bool))
        result_records.append({
            "instance_id": len(result_records),
            "representative_source_instance_id": records[representative]["instance_id"],
            "prompt_variants": [records[i]["prompt"] for i in group],
            "source_instance_ids": [records[i]["instance_id"] for i in group],
            "score": records[representative]["score"],
            "bbox_xyxy": bbox(masks[representative]),
            "mask_area_pixels": int(masks[representative].sum()),
        })
    return result_masks, result_records


def exclusive_pixel_masks(masks, image_shape=None):
    """Keep pixels covered by exactly one distinct deduplicated instance."""
    if not masks:
        return [], np.zeros(tuple(image_shape or (0, 0)), dtype=np.uint16)
    coverage = np.sum(np.asarray(masks, dtype=np.uint16), axis=0, dtype=np.uint16)
    return [mask & (coverage == 1) for mask in masks], coverage


def associate(detections, masks, instances):
    metrics = np.zeros((len(detections), len(masks), 4), dtype=np.float64)
    eligible = np.zeros((len(detections), len(masks)), dtype=bool)
    for di, detection in enumerate(detections):
        dbox = detection["bbox_xyxy"]
        darea = max(0., dbox[2] - dbox[0]) * max(0., dbox[3] - dbox[1])
        for mi, mask in enumerate(masks):
            mbox = instances[mi]["bbox_xyxy"]
            intersection = box_intersection(dbox, mbox)
            marea = max(0., mbox[2] - mbox[0]) * max(0., mbox[3] - mbox[1])
            mask_coverage = intersection / marea if marea else 0.
            detection_coverage = intersection / darea if darea else 0.
            iou = box_iou(dbox, mbox)
            # Exact raster coverage is also retained separately from tight-box coverage.
            x0, y0 = max(0, int(np.floor(dbox[0]))), max(0, int(np.floor(dbox[1])))
            x1, y1 = min(mask.shape[1], int(np.ceil(dbox[2]))), min(mask.shape[0], int(np.ceil(dbox[3])))
            raster_coverage = float(mask[y0:y1, x0:x1].sum() / max(1, int(mask.sum()))) if x1 > x0 and y1 > y0 else 0.
            metrics[di, mi] = (iou, mask_coverage, detection_coverage, raster_coverage)
            eligible[di, mi] = iou >= MIN_ASSOCIATION_BOX_IOU and raster_coverage >= MIN_ASSOCIATION_MASK_COVERAGE

    # Maximizing mean(box IoU, raster mask coverage) under one-to-one constraints.
    utility = (metrics[:, :, 0] + metrics[:, :, 3]) / 2
    rows, cols = linear_sum_assignment(-np.where(eligible, utility, -1e6)) if len(detections) and len(masks) else ([], [])
    assigned = {int(r): int(c) for r, c in zip(rows, cols) if eligible[r, c]}
    ambiguous_rows = set()
    for di, mi in assigned.items():
        alternatives = utility[di, eligible[di] & (np.arange(len(masks)) != mi)]
        if len(alternatives) and utility[di, mi] - alternatives.max() < MIN_UTILITY_MARGIN:
            ambiguous_rows.add(di)
    for di in ambiguous_rows:
        assigned.pop(di)
    results = []
    for di, detection in enumerate(detections):
        ordered = np.argsort(utility[di] * eligible[di])[::-1] if len(masks) else []
        best = assigned.get(di)
        record = {"prediction_index": int(detection["index"]), "bbox_xyxy": detection["bbox_xyxy"],
                  "status": "unmatched", "candidates": []}
        for mi in range(len(masks)):
            record["candidates"].append({"instance_id": mi, "box_iou": float(metrics[di, mi, 0]),
                                          "mask_bbox_coverage": float(metrics[di, mi, 1]),
                                          "detection_bbox_coverage": float(metrics[di, mi, 2]),
                                          "mask_pixel_coverage": float(metrics[di, mi, 3]),
                                          "eligible": bool(eligible[di, mi]),
                                          "utility": float(utility[di, mi])})
        if di in ambiguous_rows:
            record["status"] = "ignore_ambiguous_instance_match"
        elif best is not None:
            record["instance_id"] = best
            record["association_metrics"] = record["candidates"][best]
            record["status"] = "matched"
        else:
            # Distinguish geometrically plausible but one-to-one-consumed instances.
            if len(ordered) and eligible[di, ordered[0]]:
                record["status"] = "unmatched_one_to_one_conflict"
        results.append(record)
    return results, metrics, eligible


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", nargs="*", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    rows = json.loads((SOURCE / "index.json").read_text())["rows"]
    by_frame = {int(Path(row["path"]).stem): row for row in rows}
    frame_ids = sorted(by_frame) if args.frames is None else sorted(set(args.frames))
    OUT.mkdir(parents=True, exist_ok=True)
    skipped = []
    for frame in frame_ids:
        name = f"{frame:06d}"
        src_path, src_npz = SAM / f"{name}.json", SAM / f"{name}.npz"
        out_json, out_npz = OUT / f"{name}.json", OUT / f"{name}.npz"
        if not args.no_resume and out_json.is_file() and out_npz.is_file():
            existing = json.loads(out_json.read_text())
            if existing.get("source_image_sha256") == by_frame[frame].get("original_payload_sha256"):
                skipped.append(name)
                continue
        if not src_path.is_file() or not src_npz.is_file():
            raise FileNotFoundError(f"SAM instance outputs are not ready for {name}")
        src = json.loads((SAM / f"{name}.json").read_text())
        if src.get("image_source_sha256") != by_frame[frame].get("original_payload_sha256"):
            raise ValueError(f"SAM masks do not match current source image payload: {name}")
        with np.load(SAM / f"{name}.npz") as archive:
            source_masks = archive["masks"].astype(bool)
        masks, instances = deduplicate(list(source_masks), src["candidates"])
        raw_path = RAW / f"{name}.json"
        raw = json.loads(raw_path.read_text())
        if raw.get("source_payload_sha256") != by_frame[frame].get("original_payload_sha256"):
            raise ValueError(f"Raw WildDet3D outputs do not match current source image: {name}")
        detections = [d for d in raw["items"] if d["category"] == "car" and d["score"] >= DISPLAY_SCORE_THRESHOLD and d["score_3d"] >= DISPLAY_SCORE_3D_THRESHOLD]
        associations, metrics, eligible = associate(detections, masks, instances)

        # Any raster pixel covered by multiple distinct masks is ambiguous and excluded.
        image_width, image_height = src["image_size_wh"]
        exclusive_masks, cover = exclusive_pixel_masks(masks, (image_height, image_width))
        stacked = np.asarray(masks, dtype=np.uint8).reshape(-1, *cover.shape)
        exclusive = np.asarray(exclusive_masks, dtype=np.uint8).reshape(-1, *cover.shape)
        report = {
            "frame": name,
            "source_frame_id": by_frame[frame].get("frame_id"),
            "source_image_sha256": by_frame[frame].get("original_payload_sha256"),
            "sam_instance_json": str(src_path), "raw_prediction_json": str(raw_path),
            "active_display_thresholds": {"score": DISPLAY_SCORE_THRESHOLD, "score_3d": DISPLAY_SCORE_3D_THRESHOLD},
            "method": "Hungarian one-to-one matching on tight-mask-bbox IoU and raster mask pixel coverage inside the detector bbox",
            "deduplication": {"cross_prompt_only": True, "mask_iou_min": DEDUP_MASK_IOU,
                              "tight_bbox_iou_min": DEDUP_BOX_IOU,
                              "representative": "highest SAM score; no mask union; all source prompt IDs preserved"},
            "association_gates": {"tight_bbox_iou_min": MIN_ASSOCIATION_BOX_IOU,
                                  "raster_mask_pixel_coverage_min": MIN_ASSOCIATION_MASK_COVERAGE,
                                  "minimum_utility_margin_over_next_eligible_mask": MIN_UTILITY_MARGIN},
            "instances": instances,
            "associations": associations,
            "unmatched_instance_ids": sorted(set(range(len(instances))) - {a["instance_id"] for a in associations if a["status"] == "matched"}),
            "ambiguous_overlap_pixel_count": int(np.count_nonzero(cover > 1)),
            "unmatched_masks_are_negative_examples": False,
            "semantic_mask_modified": False,
        }
        npz = out_npz
        tmp_npz = OUT / f"{name}.tmp.npz"
        np.savez_compressed(tmp_npz, masks=stacked, exclusive_masks=exclusive,
                            mask_cover_count=cover,
                            instance_ids=np.arange(len(instances), dtype=np.int32),
                            bbox_iou=metrics[:, :, 0], mask_bbox_coverage=metrics[:, :, 1],
                            detection_bbox_coverage=metrics[:, :, 2], raster_mask_coverage=metrics[:, :, 3],
                            eligible=eligible)
        tmp_npz.replace(out_npz)
        report["association_npz"] = str(npz)
        tmp_json = OUT / f"{name}.tmp.json"
        tmp_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        tmp_json.replace(out_json)
        print(json.dumps({"frame": name, "detections": len(detections), "instances": len(instances),
                          "matches": sum(a["status"] == "matched" for a in associations),
                          "unmatched": len(report["unmatched_instance_ids"]),
                          "ambiguous_pixels": report["ambiguous_overlap_pixel_count"]}), flush=True)
    (OUT / "run_report.json").write_text(json.dumps({"status": "complete", "frame_count": len(frame_ids),
                                                      "skipped_valid": len(skipped),
                                                      "frames": [f"{frame:06d}" for frame in frame_ids],
                                                      "display_thresholds": {"score": DISPLAY_SCORE_THRESHOLD,
                                                                             "score_3d": DISPLAY_SCORE_3D_THRESHOLD}},
                                                     indent=2))


if __name__ == "__main__":
    main()
