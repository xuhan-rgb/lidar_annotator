"""Per-track V6 vehicle refinement with a held-out validation split."""
import math

import numpy as np

from temporal_refine import box_iou, lidar_yaw_from_world_heading, wrap_angle, world_heading
from review_bev_pose import projected_bbox
from anchor_vehicle_refine import anchor_score
from fit_vehicle_map import edge_distances, fit_history


def bundle(observations):
    """Transform supported points and sensor origins into the map frame."""
    return (
        np.concatenate([o['points'] @ o['T'][:3, :3].T + o['T'][:3, 3] for o in observations]),
        np.concatenate([np.tile(o['T'][:3, 3], (len(o['points']), 1)) for o in observations]),
        np.concatenate([np.full(len(o['points']), o['frame']) for o in observations]),
    )


def metrics(box, observations):
    means, medians, ious = [], [], []
    for o in observations:
        points = o['points'] @ o['T'][:3, :3].T + o['T'][:3, 3]
        origins = np.tile(o['T'][:3, 3], (len(points), 1))
        distances = edge_distances(points, box[:3], box[6], box[3:6], origins)
        means.append(float(np.mean(np.sqrt(1 + (distances / .15) ** 2) - 1)))
        medians.append(float(np.median(distances)))
        local = box.copy()
        local[:3] = o['T'][:3, :3].T @ (box[:3] - o['T'][:3, 3])
        local[6] = lidar_yaw_from_world_heading(box[6], o['T'])
        projected = projected_bbox(local, o['K'], o['E'], (1920, 1080))
        ious.append(box_iou(projected, o['bbox']) if projected is not None else 0.)
    if not observations:
        return dict(frame_balanced_robust_edge_cost=float('nan'), median_frame_edge_m=float('nan'),
                    median_projected_bbox_iou=float('nan'), frames=0)
    return dict(frame_balanced_robust_edge_cost=float(np.mean(means)),
                median_frame_edge_m=float(np.median(medians)),
                median_projected_bbox_iou=float(np.median(ious)), frames=len(observations))


def fit(anchor, observations, size):
    T = anchor['T']
    box = anchor['box'].copy()
    box[3:6] = size
    box[:3] = T[:3, :3] @ box[:3] + T[:3, 3]
    box[6] = world_heading(box[6], T)
    camera = anchor['E'][:3, :3] @ anchor['box'][:3] + anchor['E'][:3, 3]
    inverse = np.linalg.inv(anchor['E'])
    depth = camera[2]
    origin = T[:3, :3] @ inverse[:3, 3] + T[:3, 3]
    step = T[:3, :3] @ inverse[:3, :3] @ (camera / depth)
    points, origins, frames = bundle(observations)
    candidate, diag = fit_history(points, origins, frames, box, origin, step, depth,
                                  world_heading(anchor['initial_yaw'], T))
    assert np.linalg.norm(np.cross(candidate[:3] - origin, step)) < 1e-6
    np.testing.assert_array_equal(candidate[3:6], size)
    local = T[:3, :3].T @ (candidate[:3] - T[:3, 3])
    new_camera = anchor['E'][:3, :3] @ local + anchor['E'][:3, 3]
    original_pixel = anchor['K'] @ camera
    new_pixel = anchor['K'] @ new_camera
    error = np.linalg.norm(original_pixel[:2] / original_pixel[2] - new_pixel[:2] / new_pixel[2])
    assert error < 1e-6
    return candidate, dict(diag, anchor_frame=int(anchor['frame']), anchor_pixel_error=float(error))


def optimize_track(track_id, observations, old_box_map):
    """Return a JSON-ready V6 optimization report for one vehicle track."""
    old = np.asarray(old_box_map, dtype=float).copy()
    obs = list(observations)
    frames = [int(o['frame']) for o in obs]
    point_count = sum(len(o['points']) for o in obs)
    base = dict(track_id=int(track_id), source_frames=frames, source_point_count=int(point_count),
                source_points_deleted=0, dimensions_unchanged=True, old_box_map=old.tolist(),
                final_box_map=old.tolist(), candidate_box_map=old.tolist(), adopted=False, candidates=[])
    if len(obs) < 5:
        base.update(status='insufficient_validation', train_frames=[], validation_frames=frames,
                    before_metrics={'all': metrics(old, obs)}, after_metrics={'all': metrics(old, obs)})
        return base

    rng = np.random.default_rng(20261018)
    nval = max(1, math.ceil(0.2 * len(obs)))
    if len(obs) - nval < 3:
        base.update(status='insufficient_validation', train_frames=[], validation_frames=frames,
                    before_metrics={'all': metrics(old, obs)}, after_metrics={'all': metrics(old, obs)})
        return base
    val_indices = set(rng.choice(len(obs), nval, replace=False).tolist())
    train = [o for i, o in enumerate(obs) if i not in val_indices]
    val = [o for i, o in enumerate(obs) if i in val_indices]
    base['train_frames'] = [int(o['frame']) for o in train]
    base['validation_frames'] = [int(o['frame']) for o in val]
    old_train, old_val = metrics(old, train), metrics(old, val)
    reliable = []
    for source_index, o in enumerate(train):
        quality = anchor_score(o)
        if quality is not None:
            reliable.append((source_index, o, quality))
    reliable.sort(key=lambda item: -item[2]['score'])
    candidates = []
    for _, anchor, quality in reliable[:8]:
        candidate, diag = fit(anchor, train, old[3:6])
        candidate = np.asarray(candidate, dtype=float)
        train_metrics = metrics(candidate, train)
        candidates.append(dict(box_map=candidate.tolist(), train_metrics=train_metrics,
                               anchor_quality=quality, **diag,
                               eligible=bool(diag.get('converged', True) and
                                             train_metrics['median_projected_bbox_iou'] >=
                                             old_train['median_projected_bbox_iou'] - 0.03)))
    base['candidates'] = candidates
    base['before_metrics'] = {'train': old_train, 'validation': old_val, 'all': metrics(old, obs)}
    if not candidates:
        base.update(status='no_reliable_anchor', after_metrics=base['before_metrics'])
        return base
    eligible = [candidate for candidate in candidates if candidate['eligible']]
    if not eligible:
        base.update(status='no_eligible_candidate', after_metrics=base['before_metrics'])
        return base

    winner = min(eligible, key=lambda candidate: candidate['train_metrics']['frame_balanced_robust_edge_cost'])
    candidate = np.asarray(winner['box_map'], dtype=float)
    candidate_val = metrics(candidate, val)
    base['candidate_box_map'] = candidate.tolist()
    base['candidate_validation_metrics'] = candidate_val
    passed = bool(candidate_val['frame_balanced_robust_edge_cost'] < old_val['frame_balanced_robust_edge_cost']
                  and candidate_val['median_frame_edge_m'] < old_val['median_frame_edge_m']
                  and candidate_val['median_projected_bbox_iou'] >= old_val['median_projected_bbox_iou'] - 0.03)
    final = candidate if passed else old
    base.update(status='adopted' if passed else 'validation_failed', adopted=passed,
                final_box_map=final.tolist(), selected_anchor_frame=int(winner['anchor_frame']),
                after_metrics={'train': metrics(final, train), 'validation': metrics(final, val),
                               'all': metrics(final, obs)},
                center_shift_m=float(np.linalg.norm(final[:3] - old[:3])),
                yaw_change_deg=float(np.degrees(wrap_angle(final[6] - old[6]))))
    return base
