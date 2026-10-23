"""One centre-distance gate that preserves whole-box support and scan gaps."""
import numpy as np

from review_bev_pose import fit_box

MARGIN_METERS = 1.5
MIN_SUPPORT_POINTS = 10


class InsufficientSupportError(ValueError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        self.support_count = diagnostics['kept_support_count']
        self.radius_m = diagnostics['selection_radius_m']
        super().__init__(f'{self.support_count} points inside centre-distance radius {self.radius_m:.3f}m')


def center_distance_mask(points, center, dimensions):
    radius = float(np.linalg.norm(np.asarray(dimensions, float)/2)+MARGIN_METERS)
    distances = np.linalg.norm(np.asarray(points, float)-np.asarray(center, float), axis=1)
    return distances <= radius, radius, distances


def fit_with_center_distance(prior, points, K, E, mask_bbox, image_wh):
    """All-instance provisional fit -> one radius gate -> final fixed-pixel fit."""
    points = np.asarray(points, float)
    provisional, first = fit_box(prior, points, K=K, E=E,
                                 mask_bbox=mask_bbox, image_wh=image_wh)
    keep, radius, distances = center_distance_mask(points, provisional[:3], prior[3:6])
    selection = dict(selection_method='single 3D centre-distance gate; no clustering',
                     input_support_count=len(points), kept_support_count=int(keep.sum()),
                     removed_support_count=int((~keep).sum()),
                     selection_radius_m=radius, radius_margin_m=MARGIN_METERS,
                     radius_formula='norm(frozen_LWH/2) + 1.5m',
                     provisional_center_lidar_xyz=provisional[:3].tolist(),
                     provisional_camera_depth_m=first['final_camera_depth_m'],
                     selection_is_repeated=False,
                     selection_distance_min_m=float(distances.min()),
                     selection_distance_max_m=float(distances.max()))
    if keep.sum() < MIN_SUPPORT_POINTS:
        raise InsufficientSupportError(selection)
    final, diagnostics = fit_box(prior, points[keep], K=K, E=E,
                                 mask_bbox=mask_bbox, image_wh=image_wh)
    diagnostics.update(selection)
    diagnostics['final_center_lidar_xyz'] = final[:3].tolist()
    return final, diagnostics, keep
