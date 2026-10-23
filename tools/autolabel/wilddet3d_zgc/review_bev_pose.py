"""Single-frame centre/yaw experiment. Restricted to the three review frames."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime_paths import DATA,MODEL_CACHE,PROJECT_ROOT

import numpy as np
from scipy.optimize import least_squares, minimize

from label_geometry import boxes_to_corners_lidar


BASE = DATA / 'wilddet3d_instance_labels_20261014'
OUT = DATA / 'vehicle_failure_review_20261014' / 'pose'
FRAMES = (393, 564)
EDGES = ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7))


def rotation(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def nearest_axis_heading(angle, reference):
    """Choose the equivalent rectangle heading within +/-90 deg of reference."""
    return reference + (angle-reference+np.pi/2) % np.pi - np.pi/2


def huber(x, delta=1.):
    a = np.abs(x)
    return np.where(a < delta, .5*a*a, delta*(a-.5*delta))


def visible_edge_distances(points_xy, center, yaw, dimensions):
    """Distance to finite rectangle sides visible from the LiDAR origin.

    These are support-envelope distances, not a solid-box first-return equation.
    """
    R = rotation(yaw)
    q = (points_xy-center) @ R
    origin = -center @ R
    half = dimensions[:2]/2
    residuals = []
    for axis in (0, 1):
        side = np.sign(origin[axis])
        if abs(origin[axis]) > half[axis]:
            normal = q[:, axis] - side*half[axis]
            tangent = np.maximum(np.abs(q[:, 1-axis])-half[1-axis], 0.)
            residuals.append(np.hypot(normal, tangent))
    if not residuals:
        return np.full(len(points_xy), 10.)
    return np.min(residuals, axis=0)


def projected_bbox(box, K, E, image_wh):
    corners = boxes_to_corners_lidar(box[None])[0] @ E[:3, :3].T + E[:3, 3]
    vertices = [p for p in corners if p[2] >= .1]
    for a, b in EDGES:
        p, q = corners[a], corners[b]
        if (p[2] < .1) != (q[2] < .1):
            vertices.append(p + (.1-p[2])/(q[2]-p[2])*(q-p))
    if not vertices:
        return None
    h = np.asarray(vertices) @ K.T
    uv = h[:, :2]/h[:, 2, None]
    bbox = np.r_[uv.min(0), uv.max(0)]
    return np.clip(bbox, 0, np.tile(image_wh, 2))


def ground_plane(source, points, size):
    xyz = source['xyz'].astype(float)
    selected = ((source['point_labels'] == 0) & np.isfinite(xyz).all(1)
                & (np.linalg.norm(xyz[:, :2]-np.median(points[:, :2], axis=0), axis=1) < max(6., size[0])))
    ground = xyz[selected]
    if len(ground) < 20:
        return None, int(len(ground))
    A = np.c_[ground[:, :2], np.ones(len(ground))]
    if np.linalg.matrix_rank(A) < 3:
        return None, int(len(ground))
    initial = np.linalg.lstsq(A, ground[:, 2], rcond=None)[0]
    fit = least_squares(lambda x: A@x-ground[:, 2], initial, loss='huber', f_scale=.05)
    return fit.x, int(len(ground))


def fit_box(prior, points, plane=None, K=None, E=None, mask_bbox=None, image_wh=None):
    """Optimize camera depth/yaw on the original full camera-centre ray."""
    prior = np.asarray(prior, float)
    points = np.asarray(points, float)
    size = prior[3:6]
    extrinsic = np.eye(4) if E is None else np.asarray(E, float)
    inverse = np.linalg.inv(extrinsic)
    initial_camera = extrinsic[:3,:3]@prior[:3] + extrinsic[:3,3]
    initial_depth = float(initial_camera[2])
    if initial_depth <= .1:
        raise ValueError('The initial centre must be in front of the camera')
    camera_ray = initial_camera/initial_depth
    ray_step = inverse[:3,:3]@camera_ray
    ray_origin = inverse[:3,3]
    median = np.median(points[:, :2], axis=0)
    eigenvalues, vectors = np.linalg.eigh(np.cov(points[:, :2].T))
    principal = np.arctan2(vectors[1, -1], vectors[0, -1]) % np.pi
    # Fit physical position first. The caller rejects out-of-ROI centres;
    # clamping the optimizer to the ROI would invent an in-range label.
    points_camera = points@extrinsic[:3,:3].T + extrinsic[:3,3]
    maximum_depth = max(2*initial_depth,
                        float(np.percentile(points_camera[:,2],95)+np.linalg.norm(size)))
    bounds = [(.2, maximum_depth), (prior[6]-np.pi/2, prior[6]+np.pi/2)]

    def box_at(x):
        # All XYZ coordinates follow the camera optical ray. A ground-plane
        # override of Z would move the projected centre and is not permitted.
        center = ray_origin+x[0]*ray_step
        return np.array([*center, *size, wrap_angle(x[1])])

    def terms(x):
        box = box_at(x)
        distance = visible_edge_distances(points[:, :2], box[:2], box[6], size)
        # Cauchy influence decays for outliers; the primary target is the
        # instance returns' distance to the visible BEV rectangle boundary.
        geometry = float(np.mean(np.log1p((distance/.15)**2)))
        image = 0.
        if mask_bbox is not None:
            predicted = projected_bbox(box, K, E, image_wh)
            if predicted is None:
                image = 100.
            else:
                scale = np.tile(np.maximum(np.asarray(mask_bbox[2:])-mask_bbox[:2], 40.), 2)
                # Bounds at the viewport boundary are censored observations.
                # Clipping predicted extent makes these one-sided constraints.
                image = float(np.mean(huber((predicted-mask_bbox)/(scale*.1))))
        # Weak regularization selects among equally supported translations/axes.
        regularization = .01*float(np.sum(((box[:2]-prior[:2])/size[0])**2) + np.sin(box[6]-prior[6])**2)
        return geometry, .25*image, regularization

    def objective(x):
        return sum(terms(x))

    starts = [np.array([initial_depth, prior[6]])]
    for angle in (principal, principal+np.pi/2, prior[6]):
        angle = nearest_axis_heading(angle, prior[6])
        R = rotation(angle)
        for axis in (0, 1):
            for sign in (-1, 1):
                center = median + sign*size[axis]/2*R[:, axis]
                denominator = float(np.dot(ray_step[:2],ray_step[:2]))
                depth = (float(np.dot(center-ray_origin[:2],ray_step[:2]))/denominator
                         if denominator > 1e-12 else initial_depth)
                starts.append(np.array([depth, angle]))
    solutions = []
    for start in starts:
        start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        result = minimize(objective, start, method='Powell', bounds=bounds,
                          options={'maxiter': 150, 'xtol': 1e-5, 'ftol': 1e-6})
        if result.success and np.isfinite(result.fun):
            solutions.append(result)
    if not solutions:
        raise RuntimeError('No finite converged centre/yaw optimization')
    best = min(solutions, key=lambda r: r.fun)
    final = box_at(best.x)
    adjustment = float(wrap_angle(final[6]-prior[6]))
    cross_track_error = float(np.linalg.norm(np.cross(final[:3]-ray_origin,ray_step))/np.linalg.norm(ray_step))
    if cross_track_error > 1e-7:
        raise AssertionError('Optimized centre left the original camera optical ray')
    final_camera = extrinsic[:3,:3]@final[:3]+extrinsic[:3,3]
    if K is not None:
        initial_h = np.asarray(K)@initial_camera
        final_h = np.asarray(K)@final_camera
        initial_uv = initial_h[:2]/initial_h[2]
        final_uv = final_h[:2]/final_h[2]
        np.testing.assert_allclose(final_uv, initial_uv, atol=1e-6)
    else:
        initial_uv = final_uv = None
    if abs(adjustment) > np.pi/2 + 1e-7:
        raise AssertionError('Directed yaw changed by more than 90 degrees')
    d = visible_edge_distances(points[:, :2], final[:2], final[6], size)
    return final, dict(objective=float(best.fun), objective_terms=list(terms(best.x)),
                      prior_objective_terms=list(terms([initial_depth, prior[6]])),
                      center_constraint='full camera optical ray; projected 3D centre fixed',
                      ray_origin_lidar_xyz=ray_origin.tolist(),
                      ray_step_lidar_xyz=ray_step.tolist(),
                      initial_center_lidar_xyz=prior[:3].tolist(),
                      initial_camera_depth_m=initial_depth,
                      final_camera_depth_m=float(final_camera[2]),
                      initial_center_uv=None if initial_uv is None else initial_uv.tolist(),
                      final_center_uv=None if final_uv is None else final_uv.tolist(),
                      center_pixel_shift=None if final_uv is None else float(np.linalg.norm(final_uv-initial_uv)),
                      cross_track_error_m=cross_track_error,
                      camera_depth_bounds_m=list(bounds[0]),
                      initial_directed_yaw_deg=float(np.degrees(prior[6])),
                      final_directed_yaw_deg=float(np.degrees(final[6])),
                      signed_yaw_adjustment_deg=float(np.degrees(adjustment)),
                      maximum_yaw_adjustment_deg=90.,
                      median_visible_edge_distance_m=float(np.median(d)),
                      principal_support_axis_deg=float(np.degrees(principal)),
                      support_covariance_eigenvalues=eigenvalues.tolist(),
                      converged_starts=len(solutions),
                      optimized_parameters=['camera_depth', 'yaw'],
                      fixed_dimensions=size.tolist(),
                      ground_plane_reference=None if plane is None else plane.tolist(),
                      ground_plane_applied_to_center=False)


def main():
    (OUT/'refined/support').mkdir(parents=True, exist_ok=True)
    for frame in FRAMES:
        name = f'{frame:06d}'
        with np.load(DATA/'labels/zgc/frames'/f'{name}.npz') as a:
            source = {k:a[k] for k in a.files}
        with np.load(BASE/'refined'/f'{name}.npz') as a:
            archive = {k:a[k] for k in a.files}
        with np.load(BASE/'refined/support'/f'{name}.npz') as a:
            support = {k:a[k] for k in a.files}
        association = json.loads((BASE/'association'/f'{name}.json').read_text())
        masks = json.loads((BASE/'instance_masks'/f'{name}.json').read_text())
        meta = json.loads((BASE/'refined'/f'{name}.json').read_text())
        raw = json.loads((DATA/'wilddet3d_zgc_singleframe_20261014/raw_predictions'/f'{name}.json').read_text())
        raw_by_id = {int(r['index']):r for r in raw['items']}
        records = {r['prediction_index']:r for r in meta['all_detections']}
        boxes, reports = [], []
        for i, seed in enumerate(archive['seed_ids']):
            r = records[int(seed)]
            ids = support['point_indices'][support['offsets'][i]:support['offsets'][i+1]]
            points = source['xyz'][ids].astype(float)
            instance = association['instances'][r['sam_instance_id']]
            prior = np.asarray(r['initial_box_3d_lidar'], float)
            # The previous production schema reduced yaw modulo pi. Recover
            # the directed model axis before applying a relative angle bound.
            inverse = np.linalg.inv(source['cam_extrinsic'])
            native = np.asarray(raw_by_id[int(seed)]['corners_cam']) @ inverse[:3, :3].T
            axis = native[0]-native[2]
            prior[6] = np.arctan2(axis[1], axis[0])
            plane, ground_count = ground_plane(source, points, prior[3:6])
            box, details = fit_box(prior, points, plane, source['cam_intrinsic'], source['cam_extrinsic'],
                                   np.asarray(instance['bbox_xyxy'], float), masks['image_size_wh'])
            boxes.append(box)
            details.update(seed_id=int(seed), support_count=len(ids), ground_count=ground_count,
                           old_box=archive['box_3d'][i].tolist(), new_box=box.tolist())
            reports.append(details)
        archive['box_3d'] = np.asarray(boxes, np.float32).reshape(-1, 7)
        archive['xy'] = archive['box_3d'][:, :2].copy()
        archive['initial_directed_yaw'] = np.radians([r['initial_directed_yaw_deg'] for r in reports]).astype(np.float32)
        archive['yaw_adjustment'] = wrap_angle(archive['box_3d'][:, 6]-archive['initial_directed_yaw']).astype(np.float32)
        if np.any(np.abs(archive['yaw_adjustment']) > np.pi/2+1e-6):
            raise AssertionError('Serialized yaw exceeds relative angle bound')
        archive['corners_3d_lidar'] = boxes_to_corners_lidar(archive['box_3d']).astype(np.float32)
        np.savez_compressed(OUT/'refined'/f'{name}.npz', **archive)
        np.savez_compressed(OUT/'refined/support'/f'{name}.npz', **support)
        (OUT/'refined'/f'{name}.json').write_text(json.dumps(dict(frame=name, review_only=True,
            added_to_training=False, manual_gt=False, accepted_count=len(boxes), reports=reports,
            yaw_convention='directed model length axis, radians in [-pi,pi], shortest change <=90deg',
            objective='mean(log1p((visible_BEV_edge_distance/0.15m)^2)) + 0.25*image_extent_huber + 0.01*weak_centre_axis_regularization',
            method='single-frame instance points, joint BEV centre/yaw, fixed Mono dimensions, local ground and visible image extent'), indent=2))
        print(json.dumps(dict(frame=name, reports=reports)), flush=True)


if __name__ == '__main__':
    main()
